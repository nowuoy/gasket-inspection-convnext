from __future__ import annotations

import hashlib
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    checkpoint_safe_config,
    choose_device,
    defect_ids,
    resolve_project_path,
)
from .dataset import build_datasets, calculate_positive_weights
from .model import SingleImageConvNeXt, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.2 계열 호환
        return torch.cuda.amp.GradScaler(enabled=enabled)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    defect_loss_fn: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = batch["defect_targets"]
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("현재 batch에 유효하지 않은 결함 라벨이 있습니다.")
    loss = defect_loss_fn(output["defect_logits"], targets)
    value = float(loss.detach().item())
    return loss, {"defect_bce": value, "total": value}


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in ("image", "defect_targets"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


@torch.inference_mode()
def evaluate(
    model: SingleImageConvNeXt,
    loader: DataLoader,
    device: torch.device,
    defect_loss_fn: nn.Module,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(batch["image"])
            loss, _ = _compute_loss(output, batch, defect_loss_fn)
        batch_size = int(batch["image"].shape[0])
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("validation 데이터가 비어 있습니다.")
    return {"loss": loss_sum / sample_count}


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _save_checkpoint(
    path: Path,
    *,
    model: SingleImageConvNeXt,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_selection_value: float,
    selection_metric: str,
    selection_mode: str,
    cfg: dict[str, Any],
    manifest_hash: str,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 4,
        "task_type": "single_image_multilabel_defect_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "best_selection_value": best_selection_value,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "defect_ids": defect_ids(cfg),
        "model_config": deepcopy(cfg["model"]),
        "input_config": deepcopy(cfg["input"]),
        "full_config": checkpoint_safe_config(cfg),
        "manifest_sha256": manifest_hash,
        "metrics": metrics,
        "torch_version": torch.__version__,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train(cfg: dict[str, Any]) -> Path:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    ordered_defects = defect_ids(cfg)
    train_dataset, val_dataset, rows = build_datasets(cfg)
    train_cfg = cfg["train"]

    device = torch.device(choose_device(str(cfg.get("device", "auto"))))
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    model = build_model(len(ordered_defects), cfg["model"]).to(device)
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    model.set_backbone_trainable(freeze_epochs == 0)

    backbone_params = list(model.backbone.parameters()) + list(model.feature_norm.parameters())
    head_params = list(model.defect_score_head.parameters())
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": float(train_cfg["backbone_learning_rate"])},
            {"params": head_params, "lr": float(train_cfg["learning_rate"])},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - freeze_epochs))

    positive_weights = None
    if train_cfg.get("positive_class_weighting", "auto") == "auto":
        positive_weights = calculate_positive_weights(rows, ordered_defects).to(device)
    defect_loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights)

    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    output_dir = resolve_project_path(cfg, train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_project_path(cfg, train_cfg["manifest"])
    manifest_hash = sha256_file(manifest_path)
    history: list[dict[str, Any]] = []
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "loss"))
    checkpoint_mode = str(train_cfg.get("checkpoint_mode", "min"))
    best_selection_value = math.inf
    stale_epochs = 0
    patience = int(train_cfg.get("early_stopping_patience", 7))

    print(f"device={device}, train={len(train_dataset)}, val={len(val_dataset)}")
    for epoch in range(epochs):
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.set_backbone_trainable(True)
            print("ConvNeXt backbone fine-tuning을 시작합니다.")

        model.train()
        if epoch < freeze_epochs:
            model.backbone.eval()
            model.feature_norm.eval()
        running_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(batch["image"])
                loss, loss_parts = _compute_loss(output, batch, defect_loss_fn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_losses.append(loss_parts["total"])
            progress.set_postfix(loss=f"{loss_parts['total']:.4f}")

        if epoch >= freeze_epochs:
            scheduler.step()
        metrics = evaluate(
            model,
            val_loader,
            device,
            defect_loss_fn,
            amp_enabled,
        )
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(running_losses))
        history.append(metrics)
        print(
            f"epoch={epoch + 1} train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f}"
        )

        selection_value = metrics.get(checkpoint_metric)
        if selection_value is None or not math.isfinite(float(selection_value)):
            raise ValueError(
                f"checkpoint_metric={checkpoint_metric} 값을 계산할 수 없습니다."
            )
        selection_value = float(selection_value)
        is_best = selection_value < best_selection_value
        if is_best:
            best_selection_value = selection_value
            stale_epochs = 0
        else:
            stale_epochs += 1

        _save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_selection_value=best_selection_value,
            selection_metric=checkpoint_metric,
            selection_mode=checkpoint_mode,
            cfg=cfg,
            manifest_hash=manifest_hash,
            metrics=metrics,
        )
        if is_best:
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_selection_value=best_selection_value,
                selection_metric=checkpoint_metric,
                selection_mode=checkpoint_mode,
                cfg=cfg,
                manifest_hash=manifest_hash,
                metrics=metrics,
            )
        _json_dump(output_dir / "history.json", {"epochs": history})

        if stale_epochs >= patience:
            print(
                f"validation {checkpoint_metric}이(가) {patience}회 개선되지 않아 조기 종료합니다."
            )
            break

    return output_dir / "best.pt"
