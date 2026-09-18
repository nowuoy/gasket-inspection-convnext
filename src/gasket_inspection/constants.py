from __future__ import annotations

EXPECTED_DEFECT_IDS = (
    "shrinkage",
    "thread_defect",
    "incomplete_molding",
    "burr",
    "contamination",
)

MANIFEST_COLUMNS = (
    "sample_id",
    "image_path",
    *EXPECTED_DEFECT_IDS,
    "split",
)

FINAL_STATUSES = {"OK", "NG"}
