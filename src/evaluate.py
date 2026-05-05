from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data import LABEL_ID_TO_NAME
from src.utils import ensure_dir


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    labels = [0, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_epoch_history(history: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(history).to_csv(path, index=False)


def plot_training_curves(history: list[dict], out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    if not history:
        return
    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_accuracy"], marker="o", label="train_accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], marker="o", label="val_accuracy")
    plt.plot(df["epoch"], df["train_macro_f1"], marker="o", label="train_macro_f1")
    plt.plot(df["epoch"], df["val_macro_f1"], marker="o", label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Accuracy and Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "metric_curve.png", dpi=160)
    plt.close()


def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], [LABEL_ID_TO_NAME[0], LABEL_ID_TO_NAME[1]])
    plt.yticks([0, 1], [LABEL_ID_TO_NAME[0], LABEL_ID_TO_NAME[1]])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_metrics_summary(metrics: dict, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    json_path = out_dir / "metrics_summary.json"
    md_path = out_dir / "metrics_summary.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    lines = [
        "# Metrics Summary",
        "",
        f"- Dataset: `{metrics.get('dataset')}`",
        f"- Model: `{metrics.get('model_type')}`",
        f"- Bidirectional: `{metrics.get('bidirectional')}`",
        f"- Number of layers: `{metrics.get('num_layers')}`",
        f"- Seed: `{metrics.get('seed')}`",
        f"- Device: `{metrics.get('device')}`",
        "",
        "## Validation",
        f"- Loss: `{metrics['val']['loss']:.4f}`",
        f"- Accuracy: `{metrics['val']['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['val']['macro_f1']:.4f}`",
        "",
        "## Test",
        f"- Loss: `{metrics['test']['loss']:.4f}`",
        f"- Accuracy: `{metrics['test']['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['test']['macro_f1']:.4f}`",
        "",
        "## Interpretation prompt",
        "Điền nhận xét: mô hình có overfit không? Confusion matrix cho thấy lớp nào khó hơn? Kết quả này có tốt hơn baseline không?",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def create_baseline_vs_lab4(
    lab4_metrics: dict,
    output_path: str | Path,
    baseline_metrics_path: str | None = None,
) -> None:
    rows = []
    if baseline_metrics_path:
        try:
            baseline = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
            rows.append({
                "source": "previous_baseline",
                "model": baseline.get("model", baseline.get("model_type", "baseline")),
                "accuracy": baseline.get("test", {}).get("accuracy"),
                "macro_f1": baseline.get("test", {}).get("macro_f1"),
                "notes": "Loaded from --baseline_metrics_path",
            })
        except Exception as exc:
            rows.append({
                "source": "previous_baseline",
                "model": "could_not_read",
                "accuracy": None,
                "macro_f1": None,
                "notes": str(exc),
            })
    rows.append({
        "source": "lab4_current_run",
        "model": lab4_metrics.get("model_type"),
        "accuracy": lab4_metrics.get("test", {}).get("accuracy"),
        "macro_f1": lab4_metrics.get("test", {}).get("macro_f1"),
        "notes": "Current Lab 4 run",
    })
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)
