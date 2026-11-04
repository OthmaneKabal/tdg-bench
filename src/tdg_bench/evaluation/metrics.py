"""Classification metrics exposed by the TDGBench evaluation API."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

try:
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except ImportError as exc:  # pragma: no cover - optional dependency guard
    from tdg_bench.exceptions import OptionalDependencyError

    raise OptionalDependencyError(
        "Evaluation metrics require scikit-learn. Install: pip install 'tdg-bench[gnn]'"
    ) from exc


SUPPORTED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "matthews_corrcoef",
    "cohen_kappa",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_macro",
    "roc_auc_ovo_weighted",
    "average_precision_macro",
    "average_precision_micro",
    "average_precision_weighted",
    "log_loss",
)

METRIC_ALIASES = {
    "macro_f1": "f1_macro",
    "micro_f1": "f1_micro",
    "weighted_f1": "f1_weighted",
}


@dataclass(frozen=True)
class ClassMetrics:
    index: int
    label: str
    precision: float
    recall: float
    f1: float
    support: int


def available_metric_names() -> tuple[str, ...]:
    return SUPPORTED_METRICS


def normalize_metric_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Metric names must be strings")
    normalized = METRIC_ALIASES.get(name.strip().lower(), name.strip().lower())
    if normalized not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown metric {name!r}. Available metrics: {', '.join(SUPPORTED_METRICS)}"
        )
    return normalized


def resolve_metric_names(metrics: str | Iterable[str] | None) -> tuple[str, ...]:
    if metrics is None:
        return ("accuracy", "f1_macro")
    if isinstance(metrics, str):
        if metrics.strip().lower() == "all":
            return SUPPORTED_METRICS
        values = (metrics,)
    else:
        values = tuple(metrics)
    if not values:
        raise ValueError("metrics must not be empty")
    return tuple(dict.fromkeys(normalize_metric_name(name) for name in values))


def _safe_metric(function, *args, **kwargs) -> float:
    try:
        value = function(*args, **kwargs)
    except ValueError:
        return math.nan
    return float(value)


def calculate_metrics(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    metric_names: tuple[str, ...],
    label_names: tuple[str, ...],
    class_wise: bool,
) -> tuple[float, int, dict[str, float], tuple[ClassMetrics, ...] | None, tuple[tuple[int, ...], ...] | None]:
    support = int(mask.sum())
    if support == 0:
        raise ValueError("Cannot evaluate an empty split partition")
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    probabilities = torch.softmax(selected_logits, dim=-1).detach().cpu().numpy()
    y_true = selected_labels.detach().cpu().numpy()
    y_pred = probabilities.argmax(axis=1)
    num_classes = logits.size(1)
    classes = np.arange(num_classes)
    loss = float(torch.nn.functional.cross_entropy(selected_logits, selected_labels).item())

    values: dict[str, float] = {}
    for name in metric_names:
        if name == "accuracy":
            value = accuracy_score(y_true, y_pred)
        elif name == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, y_pred)
        elif name.startswith("precision_"):
            value = precision_score(
                y_true, y_pred, average=name.removeprefix("precision_"), zero_division=0
            )
        elif name.startswith("recall_"):
            value = recall_score(
                y_true, y_pred, average=name.removeprefix("recall_"), zero_division=0
            )
        elif name.startswith("f1_"):
            value = f1_score(
                y_true, y_pred, average=name.removeprefix("f1_"), zero_division=0
            )
        elif name == "matthews_corrcoef":
            value = matthews_corrcoef(y_true, y_pred)
        elif name == "cohen_kappa":
            value = cohen_kappa_score(y_true, y_pred)
        elif name.startswith("roc_auc_"):
            strategy, average = name.removeprefix("roc_auc_").split("_", 1)
            if num_classes == 2:
                value = _safe_metric(roc_auc_score, y_true, probabilities[:, 1])
            else:
                value = _safe_metric(
                    roc_auc_score,
                    y_true,
                    probabilities,
                    labels=classes,
                    multi_class=strategy,
                    average=average,
                )
        elif name.startswith("average_precision_"):
            average = name.removeprefix("average_precision_")
            if num_classes == 2:
                value = _safe_metric(
                    average_precision_score, y_true, probabilities[:, 1]
                )
            else:
                binary = label_binarize(y_true, classes=classes)
                value = _safe_metric(
                    average_precision_score,
                    binary,
                    probabilities,
                    average=average,
                )
        elif name == "log_loss":
            value = _safe_metric(log_loss, y_true, probabilities, labels=classes)
        else:  # pragma: no cover - resolve_metric_names prevents this branch
            raise AssertionError(name)
        values[name] = float(value)

    per_class = None
    matrix = None
    if class_wise:
        precision, recall, f1, class_support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=classes,
            zero_division=0,
        )
        per_class = tuple(
            ClassMetrics(
                index=index,
                label=label_names[index],
                precision=float(precision[index]),
                recall=float(recall[index]),
                f1=float(f1[index]),
                support=int(class_support[index]),
            )
            for index in range(num_classes)
        )
        raw_matrix = confusion_matrix(y_true, y_pred, labels=classes)
        matrix = tuple(tuple(int(value) for value in row) for row in raw_matrix)
    return loss, support, values, per_class, matrix
