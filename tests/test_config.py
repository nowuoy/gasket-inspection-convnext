from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gasket_inspection.config import (
    ConfigError,
    checkpoint_safe_config,
    defect_ids,
    load_config,
    validate_config,
    validate_decision_config,
)


def test_default_config_has_fixed_class_order() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    assert defect_ids(cfg) == [
        "shrinkage",
        "thread_defect",
        "incomplete_molding",
        "burr",
        "contamination",
    ]
    assert set(cfg["decision"]["defect_score_thresholds"]) == set(defect_ids(cfg))
    assert cfg["train"]["checkpoint_metric"] == "loss"
    assert cfg["train"]["checkpoint_mode"] == "min"
    assert "good" not in defect_ids(cfg)


def test_training_checkpoint_selection_rejects_threshold_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    cfg["train"]["checkpoint_metric"] = "macro_f1"
    cfg["train"]["checkpoint_mode"] = "max"
    with pytest.raises(ConfigError, match="validation loss"):
        validate_config(cfg)


def test_threshold_validation_is_reserved_for_watch_folder() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    invalid_for_watcher = deepcopy(cfg)
    invalid_for_watcher["decision"]["defect_score_thresholds"]["burr"] = 2.0

    validate_config(invalid_for_watcher)
    with pytest.raises(ConfigError, match="0~1"):
        validate_decision_config(invalid_for_watcher)


def test_training_checkpoint_does_not_embed_decision_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    saved = checkpoint_safe_config(cfg)
    assert "decision" not in saved
    assert "inference" not in saved
    assert "realtime" not in saved
