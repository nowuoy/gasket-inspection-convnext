from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import defect_ids, resolve_project_path
from .constants import MANIFEST_COLUMNS
from .images import build_transform, open_rgb


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    image_path: Path | None
    defect_targets: tuple[int, ...]
    split: str


def _binary_label(value: str, field_name: str, sample_id: str) -> int:
    normalized = value.strip()
    if normalized not in {"0", "1"}:
        raise ValueError(f"{sample_id}: {field_name} 라벨은 0 또는 1이어야 합니다.")
    return int(normalized)


def _optional_path(base_dir: Path, value: str) -> Path | None:
    if not value.strip():
        return None
    path = Path(value.strip()).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def read_manifest(path: str | Path, ordered_defect_ids: list[str]) -> list[ManifestRow]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"라벨 manifest를 찾을 수 없습니다: {manifest_path}")

    expected_columns = tuple(MANIFEST_COLUMNS)
    if tuple(ordered_defect_ids) != expected_columns[2 : 2 + len(ordered_defect_ids)]:
        raise ValueError("설정의 결함 클래스 순서와 manifest 규격이 다릅니다.")

    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("manifest header에 중복된 열이 있습니다.")
        missing_columns = [column for column in expected_columns if column not in fieldnames]
        if missing_columns:
            raise ValueError("manifest 필수 열이 없습니다: " + ", ".join(missing_columns))
        unknown_columns = [column for column in fieldnames if column not in expected_columns]
        if unknown_columns:
            raise ValueError("manifest에 알 수 없는 열이 있습니다: " + ", ".join(unknown_columns))

        for line_number, raw in enumerate(reader, start=2):
            missing_values = [column for column in expected_columns if raw.get(column) is None]
            if missing_values:
                raise ValueError(
                    f"manifest {line_number}행의 열 개수가 부족합니다: " + ", ".join(missing_values)
                )
            sample_id = raw["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"manifest {line_number}행: sample_id가 비어 있습니다.")
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    image_path=_optional_path(manifest_path.parent, raw["image_path"]),
                    defect_targets=tuple(
                        _binary_label(raw[defect_id], defect_id, sample_id)
                        for defect_id in ordered_defect_ids
                    ),
                    split=raw["split"].strip().lower(),
                )
            )
    return rows


def validate_rows(rows: list[ManifestRow], cfg: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("manifest에 데이터 행이 없습니다. 촬영/라벨링 후 행을 추가하세요.")

    ordered_defects = defect_ids(cfg)
    seen_ids: set[str] = set()
    positive_counts: Counter[tuple[str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    path_owners: dict[str, str] = {}

    for row in rows:
        if row.sample_id in seen_ids:
            raise ValueError(f"sample_id가 중복되었습니다: {row.sample_id}")
        seen_ids.add(row.sample_id)
        if len(row.defect_targets) != len(ordered_defects):
            raise ValueError(f"{row.sample_id}: 결함 라벨 개수가 설정과 다릅니다.")
        if any(target not in {0, 1} for target in row.defect_targets):
            raise ValueError(f"{row.sample_id}: 모든 결함 라벨은 0 또는 1이어야 합니다.")
        if row.split not in {"train", "val", "test"}:
            raise ValueError(f"{row.sample_id}: split은 train, val, test 중 하나여야 합니다.")
        if row.image_path is None:
            raise ValueError(f"{row.sample_id}: image_path가 필요합니다.")
        if not row.image_path.is_file():
            raise FileNotFoundError(f"{row.sample_id}: 이미지가 없습니다: {row.image_path}")
        path_key = str(row.image_path).casefold()
        previous = path_owners.get(path_key)
        if previous is not None:
            raise ValueError(
                f"이미지 경로가 중복 사용되었습니다: {row.image_path} "
                f"({previous}, {row.sample_id})"
            )
        path_owners[path_key] = row.sample_id
        split_counts[row.split] += 1
        for defect_id, target in zip(ordered_defects, row.defect_targets, strict=True):
            positive_counts[(row.split, defect_id)] += target

    if not {"train", "val"}.issubset(split_counts):
        raise ValueError("학습에는 train과 val split이 모두 필요합니다.")
    for split_name in ("train", "val"):
        for defect_id in ordered_defects:
            positives = positive_counts[(split_name, defect_id)]
            negatives = split_counts[split_name] - positives
            if positives == 0 or negatives == 0:
                raise ValueError(
                    f"{split_name} split의 {defect_id}에는 0/1 라벨이 모두 있어야 합니다. "
                    f"positive={positives}, negative={negatives}"
                )

    counts: dict[str, int] = {}
    for split_name in sorted(split_counts):
        counts[f"{split_name}/samples"] = split_counts[split_name]
        for defect_id in ordered_defects:
            positives = positive_counts[(split_name, defect_id)]
            counts[f"{split_name}/{defect_id}/positive"] = positives
            counts[f"{split_name}/{defect_id}/negative"] = split_counts[split_name] - positives
    return {"num_samples": len(rows), "counts": counts}


class GasketDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        cfg: dict[str, Any],
        *,
        training: bool,
    ) -> None:
        self.rows = rows
        self.input_cfg = cfg["input"]
        self.transform = build_transform(self.input_cfg, cfg["train"] if training else None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if row.image_path is None:
            raise ValueError(f"{row.sample_id}: image_path가 필요합니다.")
        image = open_rgb(row.image_path)
        return {
            "sample_id": row.sample_id,
            "image": self.transform(image),
            "defect_targets": torch.tensor(row.defect_targets, dtype=torch.float32),
        }


def build_datasets(cfg: dict[str, Any]) -> tuple[GasketDataset, GasketDataset, list[ManifestRow]]:
    manifest_path = resolve_project_path(cfg, cfg["train"]["manifest"])
    rows = read_manifest(manifest_path, defect_ids(cfg))
    validate_rows(rows, cfg)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    return (
        GasketDataset(train_rows, cfg, training=True),
        GasketDataset(val_rows, cfg, training=False),
        rows,
    )


def calculate_positive_weights(
    rows: list[ManifestRow], ordered_defect_ids: list[str]
) -> torch.Tensor:
    train_rows = [row for row in rows if row.split == "train"]
    if not train_rows:
        raise ValueError("train split이 비어 있습니다.")
    positive_counts = [
        sum(row.defect_targets[index] for row in train_rows)
        for index in range(len(ordered_defect_ids))
    ]
    missing = [
        defect_id
        for defect_id, positives in zip(ordered_defect_ids, positive_counts, strict=True)
        if positives == 0
    ]
    if missing:
        raise ValueError("train split에 positive가 없는 결함: " + ", ".join(missing))
    values = [(len(train_rows) - positives) / positives for positives in positive_counts]
    return torch.tensor(values, dtype=torch.float32)
