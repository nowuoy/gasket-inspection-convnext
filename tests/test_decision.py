from __future__ import annotations

import math

import pytest

from gasket_inspection.decision import DecisionPolicy


DEFECTS = [
    "shrinkage",
    "thread_defect",
    "incomplete_molding",
    "burr",
    "contamination",
]
NAMES = {defect_id: defect_id for defect_id in DEFECTS}


def scores(**overrides: float) -> dict[str, float]:
    values = {defect_id: 0.02 for defect_id in DEFECTS}
    values.update(overrides)
    return values


def decision_config(**overrides):
    config = {
        "criteria_version": "test-v1",
        "defect_score_thresholds": {
            "shrinkage": 0.60,
            "thread_defect": 0.60,
            "incomplete_molding": 0.60,
            "burr": 0.60,
            "contamination": 0.60,
        },
    }
    config.update(overrides)
    return config


def test_threshold_boundary_returns_ng() -> None:
    policy = DecisionPolicy(decision_config(), DEFECTS, NAMES)
    assert policy.decide(scores(burr=0.60)).status == "NG"
    minor = policy.decide(scores(burr=0.59))
    assert minor.status == "OK"
    assert minor.result_label == "양품"


def test_multiple_defects_are_reported() -> None:
    policy = DecisionPolicy(decision_config(), DEFECTS, NAMES)
    decision = policy.decide(scores(shrinkage=0.90, incomplete_molding=0.80))
    assert decision.status == "NG"
    assert decision.defect_types == ["shrinkage", "incomplete_molding"]
    assert decision.result_label == "shrinkage + incomplete_molding"


def test_nan_score_never_becomes_ok() -> None:
    policy = DecisionPolicy(decision_config(), DEFECTS, NAMES)
    invalid = scores()
    invalid["burr"] = math.nan
    with pytest.raises(FloatingPointError):
        policy.decide(invalid)


def test_missing_score_is_rejected() -> None:
    policy = DecisionPolicy(decision_config(), DEFECTS, NAMES)
    invalid = scores()
    del invalid["burr"]
    with pytest.raises(ValueError, match="누락"):
        policy.decide(invalid)


def test_scores_are_independent_and_need_not_sum_to_one() -> None:
    policy = DecisionPolicy(decision_config(), DEFECTS, NAMES)
    decision = policy.decide({defect_id: 0.90 for defect_id in DEFECTS})
    assert decision.status == "NG"
    assert decision.defect_types == DEFECTS
