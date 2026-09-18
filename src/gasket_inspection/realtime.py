from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    defect_display_names,
    defect_ids,
    resolve_project_path,
    validate_decision_config,
)
from .decision import DecisionPolicy
from .predictor import Predictor


class SingleInstanceLock:
    """동일 inspections 파일에 watcher가 둘 이상 붙는 것을 막습니다."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        if path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError(
                f"같은 inspections JSON을 사용하는 watcher가 이미 실행 중입니다: {path}"
            ) from exc

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


@dataclass(frozen=True)
class ReadyItem:
    sample_id: str
    image_path: Path

    def input_paths(self) -> list[tuple[str, Path]]:
        return [("image", self.image_path)]


class StableFileTracker:
    def __init__(self, required_checks: int) -> None:
        if required_checks < 1:
            raise ValueError("stable_checks는 1 이상이어야 합니다.")
        self.required_checks = required_checks
        self._state: dict[Path, tuple[int, int, int]] = {}

    def is_stable(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._state.pop(path, None)
            return False
        if stat.st_size <= 0:
            return False
        signature = (stat.st_size, stat.st_mtime_ns)
        previous = self._state.get(path)
        count = previous[2] + 1 if previous and previous[:2] == signature else 1
        self._state[path] = (signature[0], signature[1], count)
        return count >= self.required_checks


class InspectionJournal:
    """완료된 제품별 통합 검사 결과를 단일 JSON 파일에 누적합니다."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        if self.path.exists():
            self._read()
        else:
            self._write({"schema_version": 1, "inspections": []})

    def _read(self) -> dict[str, Any]:
        # 실행 중 파일이 아직 없거나 외부 정리로 사라졌다면 즉시 다시 만듭니다.
        if not self.path.exists():
            document = {"schema_version": 1, "inspections": []}
            self._write(document)
            return document
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"검사 기록 JSON이 손상되었습니다: {self.path}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("inspections"), list):
            raise ValueError(f"검사 기록 JSON 형식이 올바르지 않습니다: {self.path}")
        return document

    def _write(self, document: dict[str, Any]) -> None:
        temporary_path = self.path.parent / f".{self.path.name}.tmp"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self.path)

    def append(self, inspection: dict[str, Any]) -> bool:
        document = self._read()
        fingerprint = inspection["cycle_fingerprint"]
        if any(
            item.get("cycle_fingerprint") == fingerprint
            for item in document["inspections"]
            if isinstance(item, dict)
        ):
            return False

        numbers = [
            int(item.get("inspection_number", 0))
            for item in document["inspections"]
            if isinstance(item, dict)
        ]
        inspection = dict(inspection)
        inspection["inspection_number"] = max(numbers, default=0) + 1
        document["inspections"].append(inspection)
        document["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(document)
        return True

    def committed_inputs(self) -> set[tuple[str, str]]:
        committed: set[tuple[str, str]] = set()
        for inspection in self._read()["inspections"]:
            if not isinstance(inspection, dict):
                continue
            for camera_result in inspection.get("camera_results", []):
                if not isinstance(camera_result, dict):
                    continue
                sample_id = camera_result.get("sample_id")
                fingerprint = camera_result.get("input_fingerprint")
                if isinstance(sample_id, str) and isinstance(fingerprint, str):
                    committed.add((sample_id, fingerprint))
        return committed


class AtomicResultWriter:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, sample_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError(f"안전하지 않은 sample_id입니다: {sample_id}")
        return self.results_dir / f"{sample_id}.json"

    def write(self, sample_id: str, payload: dict[str, Any]) -> Path:
        final_path = self.path_for(sample_id)
        temporary_path = self.results_dir / f".{sample_id}.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, final_path)
        return final_path


class FolderMonitor:
    def __init__(
        self,
        cfg: dict[str, Any],
        predictor: Predictor,
        *,
        instance_lock: SingleInstanceLock | None = None,
    ) -> None:
        realtime_cfg = cfg["realtime"]
        validate_decision_config(cfg)
        self.cfg = cfg
        self.predictor = predictor
        configured_defects = defect_ids(cfg)
        self.policy = DecisionPolicy(
            cfg["decision"], configured_defects, defect_display_names(cfg)
        )
        self.inbox = resolve_project_path(cfg, realtime_cfg["inbox_dir"])
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.poll_interval = float(realtime_cfg.get("poll_interval_s", 0.2))
        self.retry_errors = bool(realtime_cfg.get("retry_errors", False))
        self.retry_backoff = float(realtime_cfg.get("retry_backoff_s", 5.0))
        self.max_retry_attempts = int(realtime_cfg.get("max_retry_attempts", 3))
        self.filename_pattern = re.compile(realtime_cfg["filename_regex"], flags=re.IGNORECASE)
        self.cycle_pattern = re.compile(
            realtime_cfg.get(
                "cycle_regex",
                r"^.*object_(?P<cycle>[0-9]+)_camera_(?P<camera>[1-4])$",
            ),
            flags=re.IGNORECASE,
        )
        if not {"cycle", "camera"}.issubset(self.cycle_pattern.groupindex):
            raise ValueError("realtime.cycle_regex에는 cycle과 camera 이름 그룹이 필요합니다.")
        self.camera_count = int(realtime_cfg.get("camera_count", 4))
        if self.camera_count < 1:
            raise ValueError("realtime.camera_count는 1 이상이어야 합니다.")
        self.stability = StableFileTracker(int(realtime_cfg.get("stable_checks", 3)))
        inspections_value = realtime_cfg.get("inspections_file")
        if inspections_value is None:
            legacy_state = resolve_project_path(
                cfg, realtime_cfg.get("state_db", "runtime/state/inspections.sqlite3")
            )
            inspections_file = legacy_state.with_name("inspections.json")
        else:
            inspections_file = resolve_project_path(cfg, inspections_value)
        self.instance_lock = instance_lock or SingleInstanceLock(
            inspections_file.with_suffix(inspections_file.suffix + ".lock")
        )
        self.journal = InspectionJournal(inspections_file)
        self.writer = AtomicResultWriter(resolve_project_path(cfg, realtime_cfg["results_dir"]))
        self.handled_signatures: dict[
            str, tuple[tuple[str, str, int, int, int, int], ...]
        ] = {}
        self.retry_after: dict[str, float] = {}
        self.retry_attempts: dict[str, int] = {}
        self.input_retry_attempts: dict[str, int] = {}

    def close(self) -> None:
        self.instance_lock.close()

    def _discover(self) -> list[ReadyItem]:
        candidates: dict[str, list[Path]] = {}
        for path in sorted(self.inbox.iterdir()):
            if not path.is_file():
                continue
            match = self.filename_pattern.fullmatch(path.name)
            if not match:
                continue
            candidates.setdefault(match.group("id").upper(), []).append(path)
        ready: list[ReadyItem] = []
        for sample_id, paths in candidates.items():
            if len(paths) != 1:
                print(
                    f"[ERROR] {sample_id}: 같은 ID의 이미지가 둘 이상이므로 처리하지 않습니다: {paths}",
                    file=sys.stderr,
                )
                continue
            path = paths[0]
            if self.stability.is_stable(path):
                ready.append(ReadyItem(sample_id=sample_id, image_path=path))
        return ready

    @staticmethod
    def _stat_signature(
        item: ReadyItem,
    ) -> tuple[tuple[str, str, int, int, int, int], ...]:
        signature = []
        for input_name, path in item.input_paths():
            stat = path.stat()
            signature.append(
                (input_name, str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            )
        return tuple(signature)

    @staticmethod
    def _input_metadata(
        item: ReadyItem,
    ) -> tuple[list[dict[str, Any]], str, dict[str, bytes]]:
        records: list[dict[str, Any]] = []
        snapshots: dict[str, bytes] = {}
        fingerprint_builder = hashlib.sha256()
        for input_name, path in item.input_paths():
            before = path.stat()
            file_bytes = path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            stat = path.stat()
            if (before.st_size, before.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise OSError(f"hash 계산 중 파일이 변경되었습니다: {path}")
            records.append(
                {
                    "input_name": input_name,
                    "path": str(path),
                    "sha256": file_hash,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            fingerprint_builder.update(input_name.encode("utf-8"))
            fingerprint_builder.update(str(path).encode("utf-8"))
            fingerprint_builder.update(file_hash.encode("ascii"))
            fingerprint_builder.update(str(stat.st_size).encode("ascii"))
            fingerprint_builder.update(str(stat.st_mtime_ns).encode("ascii"))
            snapshots[input_name] = file_bytes
        return records, fingerprint_builder.hexdigest(), snapshots

    def _schedule_input_retry(self, sample_id: str) -> None:
        attempt = min(self.input_retry_attempts.get(sample_id, 0) + 1, self.max_retry_attempts)
        self.input_retry_attempts[sample_id] = attempt
        delay = self.retry_backoff * (2 ** (attempt - 1))
        self.retry_after[sample_id] = time.monotonic() + delay

    def _process(self, item: ReadyItem) -> bool:
        item_start = time.perf_counter()
        if time.monotonic() < self.retry_after.get(item.sample_id, 0.0):
            return False
        try:
            stat_signature = self._stat_signature(item)
            if self.handled_signatures.get(item.sample_id) == stat_signature:
                return False
            input_records, fingerprint, snapshots = self._input_metadata(item)
            existing_path = self.writer.path_for(item.sample_id)
            if existing_path.is_file():
                try:
                    existing_result = json.loads(existing_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing_result = None
                if (
                    isinstance(existing_result, dict)
                    and existing_result.get("input_fingerprint") == fingerprint
                ):
                    self.handled_signatures[item.sample_id] = stat_signature
                    return False
            self.input_retry_attempts.pop(item.sample_id, None)
            self.retry_after.pop(item.sample_id, None)
        except (FileNotFoundError, OSError) as exc:
            # 저장 프로그램의 rename과 scan이 겹친 경우 다음 scan에서 재발견합니다.
            self._schedule_input_retry(item.sample_id)
            print(
                f"[WARN] {item.sample_id}: 입력 파일 상태 확인 실패, backoff 후 재시도합니다: {exc}",
                file=sys.stderr,
            )
            return False

        try:
            result = self.predictor.predict_bytes(
                item.sample_id,
                image_bytes=snapshots["image"],
            )
            result["decision"] = self.policy.decide(result["defect_scores"]).to_dict()
            result["inputs"] = input_records
            result["input_fingerprint"] = fingerprint
            result.setdefault("latency_ms", {})["watcher_to_prediction"] = (
                time.perf_counter() - item_start
            ) * 1000.0
        except Exception as exc:
            attempt = self.retry_attempts.get(item.sample_id, 0) + 1
            self.retry_attempts[item.sample_id] = attempt
            if self.retry_errors and attempt < self.max_retry_attempts:
                delay = self.retry_backoff * (2 ** (attempt - 1))
                self.retry_after[item.sample_id] = time.monotonic() + delay
                print(
                    f"[WARN] {item.sample_id}: 추론 실패, {delay:.1f}초 후 재시도 "
                    f"({attempt}/{self.max_retry_attempts}): {exc}",
                    file=sys.stderr,
                )
                return False
            error_result = {
                "schema_version": 2,
                "sample_id": item.sample_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "inputs": input_records,
                "input_fingerprint": fingerprint,
                "decision": {"status": "ERROR"},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            try:
                self.writer.write(item.sample_id, error_result)
            except Exception as write_exc:
                print(
                    f"[ERROR] {item.sample_id}: 오류 결과 JSON 저장 실패: {write_exc}",
                    file=sys.stderr,
                )
                self._schedule_input_retry(item.sample_id)
                return False
            try:
                print(
                    json.dumps(error_result, ensure_ascii=False, allow_nan=False),
                    file=sys.stderr,
                    flush=True,
                )
            except (BrokenPipeError, OSError, UnicodeError):
                pass
            self.handled_signatures[item.sample_id] = stat_signature
            return True

        try:
            self.writer.write(item.sample_id, result)
        except Exception as exc:
            print(f"[ERROR] {item.sample_id}: 정상 결과 JSON 저장 실패: {exc}", file=sys.stderr)
            self._schedule_input_retry(item.sample_id)
            return False

        try:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except (BrokenPipeError, OSError, UnicodeError) as exc:
            # stdout 소비자 오류는 이미 확정된 검사 결과를 바꾸지 않습니다.
            print(f"[WARN] {item.sample_id}: stdout 출력 실패: {exc}", file=sys.stderr)
        self.handled_signatures[item.sample_id] = stat_signature
        self.retry_after.pop(item.sample_id, None)
        self.retry_attempts.pop(item.sample_id, None)
        self.input_retry_attempts.pop(item.sample_id, None)
        return True

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _remove_cycle_images(self, entries: dict[int, tuple[Path, dict[str, Any]]]) -> bool:
        """네 결과가 가리키는 입력이 그대로일 때만 해당 이미지들을 삭제합니다."""
        paths_to_remove: list[Path] = []
        inbox = self.inbox.resolve()
        for _, payload in entries.values():
            inputs = payload.get("inputs")
            if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
                print("[ERROR] 결과 JSON의 inputs가 올바르지 않아 사이클을 보존합니다.", file=sys.stderr)
                return False
            record = inputs[0]
            try:
                path = Path(str(record["path"])).resolve()
                path.relative_to(inbox)
            except (KeyError, ValueError):
                print("[ERROR] inbox 밖의 입력 경로가 결과 JSON에 있어 삭제하지 않습니다.", file=sys.stderr)
                return False
            if path in paths_to_remove:
                print(f"[ERROR] 중복 입력 이미지가 있어 사이클을 보존합니다: {path}", file=sys.stderr)
                return False
            if path.exists():
                try:
                    stat = path.stat()
                    unchanged = (
                        stat.st_size == int(record["size_bytes"])
                        and stat.st_mtime_ns == int(record["mtime_ns"])
                        and self._sha256_file(path) == str(record["sha256"])
                    )
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    print(f"[ERROR] 입력 이미지 검증 실패: {path}: {exc}", file=sys.stderr)
                    return False
                if not unchanged:
                    print(f"[ERROR] 판별 후 변경된 입력 이미지는 삭제하지 않습니다: {path}", file=sys.stderr)
                    return False
                paths_to_remove.append(path)

        for path in paths_to_remove:
            try:
                path.unlink()
            except OSError as exc:
                print(f"[ERROR] 입력 이미지 삭제 실패, 다음 scan에서 재시도합니다: {path}: {exc}", file=sys.stderr)
                return False
        return True

    def _aggregate_cycle(
        self,
        cycle: str,
        entries: dict[int, tuple[Path, dict[str, Any]]],
    ) -> dict[str, Any]:
        display_names = defect_display_names(self.cfg)
        ordered_defects = defect_ids(self.cfg)
        camera_results: list[dict[str, Any]] = []
        combined_scores: dict[str, float | None] = {}
        detected_by: dict[str, list[int]] = {key: [] for key in ordered_defects}
        inference_times: list[float] = []
        watcher_times: list[float] = []
        input_mtimes_ns: list[int] = []

        fingerprint_builder = hashlib.sha256()
        has_error = False
        for camera in sorted(entries):
            _, payload = entries[camera]
            sample_id = str(payload["sample_id"])
            input_fingerprint = str(payload["input_fingerprint"])
            decision = payload.get("decision", {})
            scores = payload.get("defect_scores", {})
            status = str(decision.get("status", "ERROR"))
            has_error = has_error or status == "ERROR"
            latency = payload.get("latency_ms", {})
            if isinstance(latency, dict):
                inference_value = latency.get("inference")
                watcher_value = latency.get("watcher_to_prediction")
                if isinstance(inference_value, (int, float)):
                    inference_times.append(float(inference_value))
                if isinstance(watcher_value, (int, float)):
                    watcher_times.append(float(watcher_value))
            input_record = payload["inputs"][0]
            try:
                input_mtimes_ns.append(int(input_record["mtime_ns"]))
            except (KeyError, TypeError, ValueError):
                pass
            detected_types = [
                key for key in decision.get("defect_types", []) if key in ordered_defects
            ]
            for defect_type in detected_types:
                detected_by[defect_type].append(camera)
            fingerprint_builder.update(str(camera).encode("ascii"))
            fingerprint_builder.update(sample_id.encode("utf-8"))
            fingerprint_builder.update(input_fingerprint.encode("ascii"))
            camera_results.append(
                {
                    "camera": camera,
                    "sample_id": sample_id,
                    "status": status,
                    "defect_types": detected_types,
                    "defect_names_ko": [display_names[key] for key in detected_types],
                    "defect_scores": {
                        key: float(scores[key]) for key in ordered_defects if key in scores
                    },
                    "latency_ms": latency if isinstance(latency, dict) else {},
                    "input_fingerprint": input_fingerprint,
                    "input": input_record,
                }
            )

        for defect_type in ordered_defects:
            values = [
                item["defect_scores"][defect_type]
                for item in camera_results
                if defect_type in item["defect_scores"]
            ]
            combined_scores[defect_type] = max(values) if values else None

        combined_types = [key for key in ordered_defects if detected_by[key]]
        thresholds = self.policy.thresholds
        detected_defects = []
        for defect_type in combined_types:
            max_score = combined_scores[defect_type]
            max_score_cameras = [
                item["camera"]
                for item in camera_results
                if item["defect_scores"].get(defect_type) == max_score
            ]
            detected_defects.append(
                {
                    "defect_type": defect_type,
                    "defect_name_ko": display_names[defect_type],
                    "max_score": max_score,
                    "threshold": thresholds[defect_type],
                    "detected_cameras": detected_by[defect_type],
                    "max_score_cameras": max_score_cameras,
                }
            )

        status = "ERROR" if has_error else ("NG" if combined_types else "OK")
        completed_at = datetime.now(timezone.utc)
        completed_ns = int(completed_at.timestamp() * 1_000_000_000)
        timing_ms: dict[str, float] = {
            "inference_sum": sum(inference_times),
            "inference_slowest_camera": max(inference_times, default=0.0),
            "watcher_to_prediction_slowest_camera": max(watcher_times, default=0.0),
        }
        if input_mtimes_ns:
            timing_ms["cycle_from_first_image"] = max(
                0.0, (completed_ns - min(input_mtimes_ns)) / 1_000_000.0
            )
            timing_ms["cycle_from_last_image"] = max(
                0.0, (completed_ns - max(input_mtimes_ns)) / 1_000_000.0
            )
        return {
            "cycle_id": f"object_{cycle}",
            "cycle_fingerprint": fingerprint_builder.hexdigest(),
            "completed_at": completed_at.isoformat(),
            "camera_count": self.camera_count,
            "timing_ms": timing_ms,
            "decision": {
                "status": status,
                "result_label": (
                    "판별 오류"
                    if status == "ERROR"
                    else " + ".join(display_names[key] for key in combined_types)
                    if combined_types
                    else "양품"
                ),
                "defect_types": combined_types,
                "defect_names_ko": [display_names[key] for key in combined_types],
                "detected_defects": detected_defects,
                "class_max_scores": combined_scores,
                "applied_rule": "union_of_camera_defects",
                "criteria_version": str(self.cfg["decision"].get("criteria_version", "unknown")),
            },
            "camera_results": camera_results,
        }

    def _load_result_entries(self) -> list[tuple[Path, dict[str, Any]]]:
        entries: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(self.writer.results_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[ERROR] 결과 JSON을 읽지 못해 보존합니다: {path}: {exc}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                print(f"[ERROR] 결과 JSON 형식이 올바르지 않아 보존합니다: {path}", file=sys.stderr)
                continue
            sample_id = payload.get("sample_id")
            input_fingerprint = payload.get("input_fingerprint")
            if not isinstance(sample_id, str) or not isinstance(input_fingerprint, str):
                print(f"[ERROR] 결과 JSON 식별 정보가 없어 보존합니다: {path}", file=sys.stderr)
                continue
            entries.append((path, payload))
        return entries

    def _finalize_ready_cycles(self) -> int:
        result_entries = self._load_result_entries()
        committed = self.journal.committed_inputs()

        # 통합 기록 직후 종료된 경우 남아 있는 개별 JSON을 재시작 시 정리합니다.
        pending_entries: list[tuple[Path, dict[str, Any]]] = []
        for path, payload in result_entries:
            identity = (str(payload["sample_id"]), str(payload["input_fingerprint"]))
            if identity in committed:
                try:
                    path.unlink()
                except OSError as exc:
                    print(f"[ERROR] 기록 완료된 결과 JSON 삭제 실패: {path}: {exc}", file=sys.stderr)
                continue
            pending_entries.append((path, payload))

        cycles: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = {}
        invalid_cycles: set[str] = set()
        for path, payload in pending_entries:
            match = self.cycle_pattern.fullmatch(str(payload["sample_id"]))
            if match is None:
                continue
            cycle = match.group("cycle")
            camera = int(match.group("camera"))
            camera_entries = cycles.setdefault(cycle, {})
            if camera in camera_entries:
                invalid_cycles.add(cycle)
                print(
                    f"[ERROR] object_{cycle}에 camera_{camera} 결과가 중복되어 보존합니다.",
                    file=sys.stderr,
                )
                continue
            camera_entries[camera] = (path, payload)

        finalized = 0
        expected_cameras = set(range(1, self.camera_count + 1))
        for cycle, entries in sorted(cycles.items()):
            if cycle in invalid_cycles or set(entries) != expected_cameras:
                continue
            if not self._remove_cycle_images(entries):
                continue
            try:
                inspection = self._aggregate_cycle(cycle, entries)
                self.journal.append(inspection)
            except Exception as exc:
                # 입력 이미지는 없어도 네 개의 개별 JSON이 남아 있으므로 다음 scan에서 복구됩니다.
                print(
                    f"[ERROR] object_{cycle} 통합 기록 실패, 개별 JSON을 보존합니다: {exc}",
                    file=sys.stderr,
                )
                continue
            for result_path, payload in entries.values():
                try:
                    result_path.unlink()
                except OSError as exc:
                    print(f"[ERROR] 통합 완료 후 결과 JSON 삭제 실패: {result_path}: {exc}", file=sys.stderr)
                sample_id = str(payload["sample_id"])
                self.handled_signatures.pop(sample_id, None)
                self.retry_after.pop(sample_id, None)
                self.retry_attempts.pop(sample_id, None)
                self.input_retry_attempts.pop(sample_id, None)
            finalized += 1
            decision = inspection["decision"]
            status = decision["status"]
            if status == "NG":
                defect_summary = ", ".join(
                    f"{item['defect_name_ko']}(카메라 {','.join(map(str, item['detected_cameras']))})"
                    for item in decision["detected_defects"]
                )
                print(
                    f"[검사 완료] {inspection['cycle_id']} 판정 결과: NG | 불량: {defect_summary}",
                    flush=True,
                )
            elif status == "OK":
                print(
                    f"[검사 완료] {inspection['cycle_id']} 판정 결과: OK",
                    flush=True,
                )
            else:
                print(
                    f"[검사 완료] {inspection['cycle_id']} 판정 결과: ERROR | 판별 오류",
                    flush=True,
                )
        return finalized

    def scan_once(self) -> int:
        processed = 0
        for item in self._discover():
            processed += int(self._process(item))
        self._finalize_ready_cycles()
        return processed

    def run_forever(self) -> None:
        print(f"{self.camera_count}카메라 제품 사이클 감시 시작: {self.inbox}")
        while True:
            try:
                self.scan_once()
            except Exception as exc:
                # 일시적인 폴더/파일 오류 한 번으로 장시간 watcher가 종료되지 않게 합니다.
                print(f"[ERROR] scan 실패, 다음 주기에 재시도합니다: {exc}", file=sys.stderr)
            time.sleep(self.poll_interval)

    def run_current_files(self) -> int:
        # 안정성 검사 횟수를 충족할 만큼 현재 폴더를 반복 조회합니다.
        total = 0
        checks = int(self.cfg["realtime"].get("stable_checks", 3)) + 1
        for _ in range(checks):
            total += self.scan_once()
            time.sleep(self.poll_interval)
        return total
