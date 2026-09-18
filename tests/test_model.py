from __future__ import annotations

from gasket_inspection.model import build_model


def test_model_has_only_multilabel_defect_score_head() -> None:
    model = build_model(
        5,
        {
            "architecture": "convnext_tiny",
            "pretrained": False,
            "dropout": 0.2,
        },
    )
    assert model.defect_score_head.out_features == 5
    assert not hasattr(model, "class_head")
    assert not hasattr(model, "severity_head")
