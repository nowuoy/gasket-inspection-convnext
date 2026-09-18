from __future__ import annotations

import json

from PIL import Image

from gasket_inspection.realtime import FolderMonitor, StableFileTracker


def test_file_must_be_unchanged_for_required_checks(tmp_path) -> None:
    path = tmp_path / "P001.jpg"
    path.write_bytes(b"first")
    tracker = StableFileTracker(required_checks=3)
    assert tracker.is_stable(path) is False
    assert tracker.is_stable(path) is False
    assert tracker.is_stable(path) is True

    path.write_bytes(b"changed-and-larger")
    assert tracker.is_stable(path) is False


class DummyPredictor:
    def predict_bytes(self, sample_id, **images):
        return {
            "schema_version": 2,
            "sample_id": sample_id,
            "defect_scores": {
                "shrinkage": 0.1,
                "thread_defect": 0.1,
                "incomplete_molding": 0.1,
                "burr": 0.1,
                "contamination": 0.1,
            },
            "latency_ms": {"inference": 1.0},
        }


class CameraDefectPredictor(DummyPredictor):
    def predict_bytes(self, sample_id, **images):
        result = super().predict_bytes(sample_id, **images)
        if sample_id.endswith("CAMERA_1"):
            result["defect_scores"]["shrinkage"] = 0.9
        if sample_id.endswith("CAMERA_2"):
            result["defect_scores"]["burr"] = 0.8
        if sample_id.endswith("CAMERA_4"):
            result["defect_scores"]["shrinkage"] = 0.7
            result["defect_scores"]["contamination"] = 0.6
        return result


def monitor_config(tmp_path, stable_checks=3):
    return {
        "_project_root": str(tmp_path),
        "input": {},
        "defect_classes": [
            {"id": "shrinkage", "name_ko": "수축불량"},
            {"id": "thread_defect", "name_ko": "나사선 불량"},
            {"id": "incomplete_molding", "name_ko": "미성형 불량"},
            {"id": "burr", "name_ko": "burr 불량"},
            {"id": "contamination", "name_ko": "이염 불량"},
        ],
        "decision": {
            "criteria_version": "test-v1",
            "defect_score_thresholds": {
                "shrinkage": 0.5,
                "thread_defect": 0.5,
                "incomplete_molding": 0.5,
                "burr": 0.5,
                "contamination": 0.5,
            },
        },
        "realtime": {
            "inbox_dir": "inbox",
            "results_dir": "results",
            "inspections_file": "state/inspections.json",
            "camera_count": 4,
            "poll_interval_s": 0.001,
            "stable_checks": stable_checks,
            "retry_errors": False,
            "retry_backoff_s": 0.01,
            "max_retry_attempts": 2,
            "filename_regex": r"^(?P<id>[A-Za-z0-9_-]+)\.(?P<ext>jpg|jpeg|png)$",
            "cycle_regex": r"^.*object_(?P<cycle>[0-9]+)_camera_(?P<camera>[1-4])$",
        },
    }


def test_run_current_files_processes_stable_image(tmp_path) -> None:
    cfg = monitor_config(tmp_path)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    Image.new("RGB", (16, 16), "yellow").save(inbox / "P001.jpg")
    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        assert monitor.run_current_files() == 1
        result_path = tmp_path / "results" / "P001.json"
        assert result_path.is_file()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["decision"]["status"] == "OK"
        assert result["decision"]["criteria_version"] == "test-v1"
    finally:
        monitor.close()


def test_watch_folder_applies_threshold_to_predictor_scores(tmp_path) -> None:
    cfg = monitor_config(tmp_path, stable_checks=1)
    cfg["decision"]["defect_score_thresholds"]["shrinkage"] = 0.05
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    Image.new("RGB", (16, 16), "yellow").save(inbox / "P003.jpg")

    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        assert monitor.scan_once() == 1
        result = json.loads(
            (tmp_path / "results" / "P003.json").read_text(encoding="utf-8")
        )
        assert result["decision"]["status"] == "NG"
        assert result["decision"]["defect_types"] == ["shrinkage"]
    finally:
        monitor.close()


def test_duplicate_id_is_not_processed(tmp_path) -> None:
    cfg = monitor_config(tmp_path, stable_checks=1)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    image = Image.new("RGB", (16, 16), "yellow")
    image.save(inbox / "P002.jpg")
    image.save(inbox / "P002.png")
    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        assert monitor.scan_once() == 0
        assert not (tmp_path / "results" / "P002.json").exists()
    finally:
        monitor.close()


def save_camera_cycle(inbox, cycle_number):
    for camera in range(1, 5):
        Image.new("RGB", (16, 16), (camera * 20, cycle_number, 0)).save(
            inbox / f"capture_object_{cycle_number:04d}_camera_{camera}.png"
        )


def test_four_camera_cycle_is_aggregated_and_source_files_are_removed(
    tmp_path, capsys
) -> None:
    cfg = monitor_config(tmp_path, stable_checks=1)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    save_camera_cycle(inbox, 1)

    monitor = FolderMonitor(cfg, CameraDefectPredictor())
    try:
        assert monitor.scan_once() == 4
    finally:
        monitor.close()

    assert list(inbox.iterdir()) == []
    assert list((tmp_path / "results").iterdir()) == []

    state = json.loads((tmp_path / "state" / "inspections.json").read_text(encoding="utf-8"))
    assert len(state["inspections"]) == 1
    inspection = state["inspections"][0]
    assert inspection["inspection_number"] == 1
    assert inspection["cycle_id"] == "object_0001"
    assert inspection["decision"]["status"] == "NG"
    assert inspection["decision"]["defect_types"] == [
        "shrinkage",
        "burr",
        "contamination",
    ]
    detected = {
        item["defect_type"]: item["detected_cameras"]
        for item in inspection["decision"]["detected_defects"]
    }
    assert detected == {"shrinkage": [1, 4], "burr": [2], "contamination": [4]}
    assert len(inspection["camera_results"]) == 4
    assert inspection["timing_ms"]["inference_sum"] == 4.0
    assert inspection["timing_ms"]["inference_slowest_camera"] == 1.0
    assert inspection["timing_ms"]["cycle_from_first_image"] >= 0.0
    assert inspection["timing_ms"]["cycle_from_last_image"] >= 0.0
    assert all("latency_ms" in item for item in inspection["camera_results"])
    terminal_output = capsys.readouterr().out
    assert "[검사 완료] object_0001 판정 결과: NG" in terminal_output
    assert "수축불량(카메라 1,4)" in terminal_output


def test_completed_cycles_accumulate_in_one_inspections_file(tmp_path) -> None:
    cfg = monitor_config(tmp_path, stable_checks=1)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        save_camera_cycle(inbox, 1)
        assert monitor.scan_once() == 4
        save_camera_cycle(inbox, 2)
        assert monitor.scan_once() == 4
    finally:
        monitor.close()

    state = json.loads((tmp_path / "state" / "inspections.json").read_text(encoding="utf-8"))
    assert [item["inspection_number"] for item in state["inspections"]] == [1, 2]
    assert [item["cycle_id"] for item in state["inspections"]] == [
        "object_0001",
        "object_0002",
    ]
    assert all(item["decision"]["status"] == "OK" for item in state["inspections"])
    assert list(inbox.iterdir()) == []
    assert list((tmp_path / "results").iterdir()) == []


def test_incomplete_camera_cycle_is_preserved_until_fourth_result(tmp_path) -> None:
    cfg = monitor_config(tmp_path, stable_checks=1)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for camera in range(1, 4):
        Image.new("RGB", (16, 16), "yellow").save(
            inbox / f"capture_object_0001_camera_{camera}.png"
        )

    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        assert monitor.scan_once() == 3
        assert len(list(inbox.glob("*.png"))) == 3
        assert len(list((tmp_path / "results").glob("*.json"))) == 3
        state = json.loads(
            (tmp_path / "state" / "inspections.json").read_text(encoding="utf-8")
        )
        assert state["inspections"] == []

        Image.new("RGB", (16, 16), "yellow").save(
            inbox / "capture_object_0001_camera_4.png"
        )
        assert monitor.scan_once() == 1
    finally:
        monitor.close()

    assert list(inbox.iterdir()) == []
    assert list((tmp_path / "results").iterdir()) == []
    state = json.loads((tmp_path / "state" / "inspections.json").read_text(encoding="utf-8"))
    assert len(state["inspections"]) == 1


def test_reused_cycle_number_with_new_images_adds_new_history_record(tmp_path) -> None:
    cfg = monitor_config(tmp_path, stable_checks=1)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        save_camera_cycle(inbox, 1)
        assert monitor.scan_once() == 4
        for camera in range(1, 5):
            Image.new("RGB", (16, 16), (camera * 20, 99, 0)).save(
                inbox / f"capture_object_0001_camera_{camera}.png"
            )
        assert monitor.scan_once() == 4
    finally:
        monitor.close()

    state = json.loads((tmp_path / "state" / "inspections.json").read_text(encoding="utf-8"))
    assert [item["inspection_number"] for item in state["inspections"]] == [1, 2]
    assert [item["cycle_id"] for item in state["inspections"]] == [
        "object_0001",
        "object_0001",
    ]
    assert state["inspections"][0]["cycle_fingerprint"] != state["inspections"][1][
        "cycle_fingerprint"
    ]
