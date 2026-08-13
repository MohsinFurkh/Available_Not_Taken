"""Training and inference for one (regime, backbone, fold, seed) configuration.

Everything that could differ between regimes and must not -- optimiser,
schedule, augmentation, epoch budget, early-stopping rule -- is fixed here and
read from one config, so that the only difference between R-naive and R-clean is
which variant array the dataset reads (proposal Section C.6).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..eval.metrics import PRIMARY_METRIC, evaluate, softmax
from ..models.backbones import AuditModel, build_model
from ..utils import get_logger, set_seed

LOGGER = get_logger("asa.train")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    model: AuditModel
    best_epoch: int
    best_val_score: float
    history: list[dict[str, Any]] = field(default_factory=list)
    train_seconds: float = 0.0
    n_parameters: int = 0
    n_trainable: int = 0


def _build_optimizer(model: nn.Module, cfg: ExperimentConfig):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    return torch.optim.SGD(parameters, lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)


def _build_scheduler(optimizer, cfg: ExperimentConfig, steps_per_epoch: int):
    if cfg.train.scheduler != "cosine":
        return None
    total = max(1, cfg.train.epochs * steps_per_epoch)
    warmup = max(1, cfg.train.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_task + lambda * L_aux. No custom loss is proposed (Section C.7):
    a bespoke loss would signal the contribution had drifted toward method
    invention."""
    target = batch["target"].to(device)
    if cfg.dataset.task == "regression":
        task_loss = nn.functional.l1_loss(outputs["logits"].squeeze(-1), target.float())
    else:
        task_loss = nn.functional.cross_entropy(outputs["logits"], target.long())

    parts = {"task": float(task_loss.detach())}
    aux_total = torch.zeros((), device=device)
    if cfg.aux.enabled:
        for name in cfg.aux.categorical_fields:
            key = f"aux_{name}"
            if key in outputs and key in batch:
                loss = nn.functional.cross_entropy(outputs[key], batch[key].to(device).long())
                aux_total = aux_total + loss
                parts[key] = float(loss.detach())
        for name in cfg.aux.numeric_fields:
            key = f"aux_{name}"
            if key in outputs and key in batch:
                loss = nn.functional.l1_loss(outputs[key], batch[key].to(device).float())
                aux_total = aux_total + loss
                parts[key] = float(loss.detach())

    total = task_loss + cfg.aux.lambda_aux * aux_total
    parts["total"] = float(total.detach())
    return total, parts


def train_one(
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    seed: int,
    aux_categorical: dict[str, int] | None = None,
    aux_numeric: list[str] | None = None,
    device: torch.device | None = None,
) -> TrainResult:
    set_seed(seed)
    device = device or get_device()
    model = build_model(cfg.model, num_classes, cfg.dataset.task, aux_categorical, aux_numeric).to(device)

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, max(1, len(train_loader)))
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    primary = PRIMARY_METRIC[cfg.dataset.task]
    higher_is_better = cfg.dataset.task == "classification"
    best_score = -np.inf if higher_is_better else np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss, parts = _losses(outputs, batch, cfg, device)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            # AMP skips the optimizer step when the scaler sees inf/nan, which
            # happens routinely on the first iterations while the scale
            # calibrates. Advancing the LR schedule for an update that did not
            # happen desynchronises the schedule from the optimiser, so the
            # scheduler only steps when the optimiser did.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
            if scheduler is not None and optimizer_stepped:
                scheduler.step()
            epoch_losses.append(parts["total"])

        val = predict(model, val_loader, cfg, device)
        val_metrics = evaluate(val["targets"], val["outputs"], cfg.dataset.task, cfg.eval.calibration_bins)
        score = val_metrics[primary]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                f"val_{primary}": float(score),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = score > best_score if higher_is_better else score < best_score
        if improved:
            best_score, best_epoch = float(score), epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.train.early_stop_patience:
            LOGGER.info("  early stop at epoch %d (best %s=%.4f @ %d)", epoch, primary, best_score, best_epoch)
            break

    # Model selection is on validation folds only; the test data is touched once
    # per configuration, after this point (Section 9).
    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        best_val_score=float(best_score),
        history=history,
        train_seconds=time.time() - started,
        n_parameters=model.n_parameters,
        n_trainable=model.n_trainable,
    )


@torch.no_grad()
def predict(
    model: AuditModel,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    device = device or get_device()
    model.eval()
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    use_amp = cfg.train.amp and device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)["logits"]
        logits = logits.float().cpu().numpy()
        outputs.append(logits)
        targets.append(batch["target"].numpy())
        rows.append(batch["row"].numpy())

    stacked = np.concatenate(outputs, axis=0)
    if cfg.dataset.task == "classification":
        stacked = softmax(stacked)
    return {
        "outputs": stacked,
        "targets": np.concatenate(targets, axis=0),
        "rows": np.concatenate(rows, axis=0),
    }
