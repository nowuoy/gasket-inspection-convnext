from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectedDefect:
    defect_type: str
    defect_name_ko: str
    defect_score: float
    threshold: float


@dataclass(frozen=True)
class InspectionDecision:
    status: str
    result_label: str
    defect_types: list[str]
    defect_names_ko: list[str]
    detected_defects: list[DetectedDefect]
    max_defect_score: float
    applied_rule: str
    evaluated_thresholds: dict[str, float]
    criteria_version: str
    provisional: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionPolicy:
    """독립적인 결함 점수를 현장 OK/NG 기준과 비교합니다."""

    def __init__(
        self,
        decision_cfg: dict[str, Any],
        ordered_defect_ids: list[str],
        display_names: dict[str, str],
    ) -> None:
        if not ordered_defect_ids:
            raise ValueError("결함 클래스가 하나 이상 필요합니다.")
        self.cfg = decision_cfg
        self.defect_ids = ordered_defect_ids
        self.display_names = display_names
        thresholds = decision_cfg.get("defect_score_thresholds", {})
        missing = set(ordered_defect_ids) - set(thresholds)
        if missing:
            raise ValueError("threshold가 누락된 결함: " + ", ".join(sorted(missing)))
        self.thresholds = {
            defect_id: float(thresholds[defect_id]) for defect_id in ordered_defect_ids
        }
        invalid = [
            defect_id
            for defect_id, threshold in self.thresholds.items()
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
        ]
        if invalid:
            raise ValueError("유효하지 않은 defect_score threshold: " + ", ".join(invalid))

    def decide(self, defect_scores: dict[str, float]) -> InspectionDecision:
        missing = set(self.defect_ids) - set(defect_scores)
        if missing:
            raise ValueError("점수가 누락된 결함: " + ", ".join(sorted(missing)))
        unknown = set(defect_scores) - set(self.defect_ids)
        if unknown:
            raise ValueError("알 수 없는 결함 점수: " + ", ".join(sorted(unknown)))
        invalid = [
            defect_id
            for defect_id in self.defect_ids
            if not math.isfinite(float(defect_scores[defect_id]))
            or not 0.0 <= float(defect_scores[defect_id]) <= 1.0
        ]
        if invalid:
            raise FloatingPointError("유효하지 않은 defect_score: " + ", ".join(invalid))

        detected = [
            DetectedDefect(
                defect_type=defect_id,
                defect_name_ko=self.display_names[defect_id],
                defect_score=float(defect_scores[defect_id]),
                threshold=self.thresholds[defect_id],
            )
            for defect_id in self.defect_ids
            if float(defect_scores[defect_id]) >= self.thresholds[defect_id]
        ]
        defect_types = [item.defect_type for item in detected]
        defect_names = [item.defect_name_ko for item in detected]
        provisional = str(self.cfg.get("criteria_version", "TODO")).upper().startswith("TODO")

        return InspectionDecision(
            status="NG" if detected else "OK",
            result_label=" + ".join(defect_names) if detected else "양품",
            defect_types=defect_types,
            defect_names_ko=defect_names,
            detected_defects=detected,
            max_defect_score=max(float(defect_scores[key]) for key in self.defect_ids),
            applied_rule="per_class_defect_score_threshold",
            evaluated_thresholds=dict(self.thresholds),
            criteria_version=str(self.cfg.get("criteria_version", "unknown")),
            provisional=provisional,
        )
