from __future__ import annotations

import pytest

from gasket_inspection.dataset import read_manifest


DEFECTS = [
    "shrinkage",
    "thread_defect",
    "incomplete_molding",
    "burr",
    "contamination",
]
HEADER = (
    "sample_id,image_path,shrinkage,thread_defect,incomplete_molding,burr,"
    "contamination,split\n"
)


def test_manifest_reads_multilabel_binary_targets(tmp_path) -> None:
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        HEADER + "P001,image.jpg,1,0,1,0,0,train\n",
        encoding="utf-8",
    )
    rows = read_manifest(manifest, DEFECTS)
    assert rows[0].defect_targets == (1, 0, 1, 0, 0)


def test_manifest_rejects_subjective_fractional_label(tmp_path) -> None:
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        HEADER + "P001,image.jpg,0.3,0,0,0,0,train\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="0 또는 1"):
        read_manifest(manifest, DEFECTS)
