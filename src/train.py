from __future__ import annotations

from copy import deepcopy
from typing import Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.evaluate import compute_classification_metrics


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["length"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item()) * batch["label"].size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(batch["label"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, compute_classification_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _move_batch(batch, device)
        logits = model(batch["input_ids"], batch["length"])
        loss = criterion(logits, batch["label"])
        total_loss += float(loss.item()) * batch["label"].size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(batch["label"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, compute_classification_metrics(y_true, y_pred)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    min_delta: float,
    epoch_logger: Callable[[dict], None] | None = None,
) -> tuple[list[dict], dict | None]:
    history: list[dict] = []
    best_state = None
    best_score = -1.0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        if epoch_logger is not None:
            epoch_logger(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_score + min_delta:
            best_score = val_metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    return history, best_state


@torch.no_grad()
def predict_with_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, list[int], list[int], list[float]]:
    model.eval()
    rows = []
    y_true, y_pred, y_prob_positive = [], [], []
    softmax = nn.Softmax(dim=1)
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = _move_batch(batch, device)
        logits = model(batch["input_ids"], batch["length"])
        probs = softmax(logits)
        preds = probs.argmax(dim=1)
        for idx, true_label, pred_label, prob_row in zip(
            batch["index"].detach().cpu().tolist(),
            batch["label"].detach().cpu().tolist(),
            preds.detach().cpu().tolist(),
            probs.detach().cpu().tolist(),
        ):
            rows.append({
                "row_index": idx,
                "true_label_id": int(true_label),
                "pred_label_id": int(pred_label),
                "prob_negative": float(prob_row[0]),
                "prob_positive": float(prob_row[1]),
                "confidence": float(max(prob_row)),
            })
            y_true.append(int(true_label))
            y_pred.append(int(pred_label))
            y_prob_positive.append(float(prob_row[1]))
    return pd.DataFrame(rows), y_true, y_pred, y_prob_positive
