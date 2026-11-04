"""Unified hidden-size, split and training-seed evaluation protocol."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tdg_bench.data_preparation import replace_bundle_split
from tdg_bench.evaluation.metrics import normalize_metric_name, resolve_metric_names
from tdg_bench.evaluation.runner import (
    EvaluationResult,
    evaluate_bundle,
    evaluation_result_from_dict,
)


DEFAULT_HIDDEN_DIMS = (64, 128, 256, 512)
DEFAULT_TRAINING_SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class SplitMetricSummary:
    mean: float
    std_seed: float
    n_seeds: int


@dataclass(frozen=True)
class VarianceSummary:
    """Separate between-split and within-split training-seed variability."""

    mean: float
    std_split: float
    std_seed: float
    std_total: float
    n_splits: int
    n_seeds_per_split: int
    n_runs: int
    by_split: dict[int, SplitMetricSummary]


@dataclass(frozen=True)
class HiddenDimensionSummary:
    hidden_dim: int
    train: dict[str, VarianceSummary]
    validation: dict[str, VarianceSummary]
    test: dict[str, VarianceSummary]
    class_wise: dict[
        str, dict[str, dict[str, VarianceSummary]]
    ] | None


@dataclass(frozen=True)
class UnifiedEvaluationResult:
    dataset: str
    graph: str
    model: str
    initial_embedding_model: str | None
    hidden_dims: tuple[int, ...]
    split_seeds: tuple[int, ...]
    training_seeds: tuple[int, ...]
    feature_seed: int
    metric_names: tuple[str, ...]
    selection_metric: str
    selected_hidden_dim: int
    results_dir: Path | None
    export_dir: Path | None
    resumed_runs: int
    completed_runs: int
    runs: tuple[EvaluationResult, ...]
    by_hidden_dim: dict[int, HiddenDimensionSummary]

    @property
    def selected_summary(self) -> HiddenDimensionSummary:
        return self.by_hidden_dim[self.selected_hidden_dim]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["results_dir"] = (
            None if self.results_dir is None else str(self.results_dir)
        )
        result["export_dir"] = (
            None if self.export_dir is None else str(self.export_dir)
        )
        result["selected_summary"] = asdict(self.selected_summary)
        return result

    def summary_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result.pop("runs", None)
        return result

    def runs_dataframe(self):
        return _dataframe(_run_rows(self))

    def summary_dataframe(self):
        return _summary_dataframe(self)

    def class_metrics_dataframe(self):
        return _dataframe(_class_metric_rows(self))

    def training_history_dataframe(self):
        return _dataframe(_training_history_rows(self))


@dataclass(frozen=True)
class UnifiedEvaluationSuite:
    dataset: str
    graph: str
    models: tuple[str, ...]
    initial_embedding_model: str | None
    selection_metric: str
    selected_model: str
    results_dir: Path | None
    resumed_runs: int
    completed_runs: int
    results: dict[str, UnifiedEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "graph": self.graph,
            "models": self.models,
            "initial_embedding_model": self.initial_embedding_model,
            "selection_metric": self.selection_metric,
            "selected_model": self.selected_model,
            "results_dir": (
                None if self.results_dir is None else str(self.results_dir)
            ),
            "resumed_runs": self.resumed_runs,
            "completed_runs": self.completed_runs,
            "results": {
                model: result.to_dict() for model, result in self.results.items()
            },
        }

    def summary_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result["results"] = {
            model: evaluation.summary_dict()
            for model, evaluation in self.results.items()
        }
        return result


def _dataframe(rows: list[dict[str, Any]]):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - broken installation guard
        from tdg_bench.exceptions import OptionalDependencyError

        raise OptionalDependencyError(
            "DataFrame exports require pandas. "
            "Reinstall with: pip install --upgrade tdg-bench"
        ) from exc
    return pd.DataFrame(rows)


def _run_rows(result: UnifiedEvaluationResult) -> list[dict[str, Any]]:
    rows = []
    runs_per_hidden = len(result.split_seeds) * len(result.training_seeds)
    for index, run in enumerate(result.runs):
        hidden_dim = result.hidden_dims[index // runs_per_hidden]
        row: dict[str, Any] = {
            "dataset": run.dataset,
            "graph": run.graph,
            "variant": run.variant,
            "model": run.model,
            "initial_embedding_model": run.initial_embedding_model or "random",
            "hidden_dim": hidden_dim,
            "split_seed": run.split_seed,
            "training_seed": run.random_seed,
            "best_epoch": run.best_epoch,
            "epochs_ran": run.epochs_ran,
            "training_mode": run.training_mode,
            "batch_size": run.batch_size,
            "num_neighbors": json.dumps(run.num_neighbors),
        }
        for partition in ("train", "validation", "test"):
            for metric, value in _global_metric_values(
                getattr(run, partition)
            ).items():
                row[f"{partition}_{metric}"] = value
        rows.append(row)
    return rows


def _summary_rows(result: UnifiedEvaluationResult) -> list[dict[str, Any]]:
    rows = []
    for hidden_dim, hidden_summary in sorted(result.by_hidden_dim.items()):
        for partition in ("train", "validation", "test"):
            for metric, summary in sorted(
                getattr(hidden_summary, partition).items()
            ):
                rows.append(
                    {
                        "dataset": result.dataset,
                        "graph": result.graph,
                        "model": result.model,
                        "hidden_dim": hidden_dim,
                        "selected_hidden_dim": result.selected_hidden_dim,
                        "partition": partition,
                        "metric": metric,
                        "mean": summary.mean,
                        "std_split": summary.std_split,
                        "std_seed": summary.std_seed,
                        "std_total": summary.std_total,
                        "n_splits": summary.n_splits,
                        "n_seeds_per_split": summary.n_seeds_per_split,
                        "n_runs": summary.n_runs,
                    }
                )
    return rows


def _summary_dataframe(result: UnifiedEvaluationResult):
    return _dataframe(_summary_rows(result))


def _class_metric_rows(result: UnifiedEvaluationResult) -> list[dict[str, Any]]:
    rows = []
    runs_per_hidden = len(result.split_seeds) * len(result.training_seeds)
    for index, run in enumerate(result.runs):
        hidden_dim = result.hidden_dims[index // runs_per_hidden]
        for partition in ("train", "validation", "test"):
            class_rows = getattr(run, partition).per_class or ()
            for metric in class_rows:
                rows.append(
                    {
                        "dataset": run.dataset,
                        "graph": run.graph,
                        "model": run.model,
                        "hidden_dim": hidden_dim,
                        "split_seed": run.split_seed,
                        "training_seed": run.random_seed,
                        "partition": partition,
                        "class_index": metric.index,
                        "class_label": metric.label,
                        "precision": metric.precision,
                        "recall": metric.recall,
                        "f1": metric.f1,
                        "support": metric.support,
                    }
                )
    return rows


def _training_history_rows(result: UnifiedEvaluationResult) -> list[dict[str, Any]]:
    rows = []
    runs_per_hidden = len(result.split_seeds) * len(result.training_seeds)
    for index, run in enumerate(result.runs):
        hidden_dim = result.hidden_dims[index // runs_per_hidden]
        for epoch in run.history:
            row: dict[str, Any] = {
                "dataset": run.dataset,
                "graph": run.graph,
                "model": run.model,
                "hidden_dim": hidden_dim,
                "split_seed": run.split_seed,
                "training_seed": run.random_seed,
                "epoch": epoch.epoch,
                "train_loss": epoch.train_loss,
                "train_accuracy": epoch.train_accuracy,
                "train_f1_macro": epoch.train_macro_f1,
                "validation_loss": epoch.validation_loss,
                "validation_accuracy": epoch.validation_accuracy,
                "validation_f1_macro": epoch.validation_macro_f1,
                "selection_metric": epoch.selection_metric,
                "selection_score": epoch.selection_score,
            }
            for metric, value in (epoch.train_values or {}).items():
                row[f"train_{metric}"] = value
            for metric, value in (epoch.validation_values or {}).items():
                row[f"validation_{metric}"] = value
            rows.append(row)
    return rows


def _confusion_matrices(result: UnifiedEvaluationResult) -> list[dict[str, Any]]:
    rows = []
    runs_per_hidden = len(result.split_seeds) * len(result.training_seeds)
    for index, run in enumerate(result.runs):
        hidden_dim = result.hidden_dims[index // runs_per_hidden]
        for partition in ("train", "validation", "test"):
            matrix = getattr(run, partition).confusion_matrix
            if matrix is not None:
                rows.append(
                    {
                        "hidden_dim": hidden_dim,
                        "split_seed": run.split_seed,
                        "training_seed": run.random_seed,
                        "partition": partition,
                        "matrix": matrix,
                    }
                )
    return rows


def _atomic_csv_write(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fallback_fields: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fallback_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@dataclass(frozen=True)
class _PreparedUnifiedData:
    base_bundle: Any
    bundles: dict[int, Any]
    split_seeds: tuple[int, ...]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=repr)
    temporary.replace(path)


def _graph_identity(bundle) -> str:
    if bundle.artifact is None:
        return _fingerprint(
            {
                "dataset": bundle.dataset,
                "terms": bundle.gold_standard.terms,
                "labels": bundle.gold_standard.labels,
            }
        )
    asset = bundle.artifact.asset
    if asset.sha256 is not None:
        return asset.sha256
    stat = bundle.artifact.path.stat()
    return _fingerprint(
        {
            "path": str(bundle.artifact.path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )


def _resolve_results_root(benchmark, results_dir: str | Path | None) -> Path:
    if results_dir is None:
        return (benchmark.data_root.parent / "tdgbench_results").resolve()
    return Path(results_dir).expanduser().resolve()


def _run_checkpoint(
    *,
    results_root: Path,
    dataset: str,
    graph: str,
    initial_embedding_model: str | None,
    model: str,
    hidden_dim: int,
    split_seed: int,
    training_seed: int,
    fingerprint: str,
) -> Path:
    initial_label = initial_embedding_model or "random"
    return (
        results_root
        / _slug(dataset)
        / _slug(graph)
        / _slug(initial_label)
        / _slug(model)
        / f"hidden_{hidden_dim}"
        / f"split_{split_seed}"
        / f"training_seed_{training_seed}-{fingerprint[:16]}.json"
    )


def _load_checkpoint(path: Path, expected_fingerprint: str) -> EvaluationResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid evaluation checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported evaluation checkpoint: {path}")
    if payload.get("protocol_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Checkpoint protocol mismatch: {path}")
    try:
        return evaluation_result_from_dict(payload["result"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Incomplete evaluation checkpoint {path}: {exc}") from exc


def _validate_unique_positive(values: Iterable[int], label: str) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(value, int) or value < 1 for value in resolved):
        raise ValueError(f"{label} must contain positive integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} must not contain duplicates")
    return resolved


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate_observations(
    observations: list[tuple[int, int, float]],
) -> VarianceSummary:
    finite = [item for item in observations if math.isfinite(item[2])]
    if not finite:
        return VarianceSummary(
            mean=math.nan,
            std_split=math.nan,
            std_seed=math.nan,
            std_total=math.nan,
            n_splits=0,
            n_seeds_per_split=0,
            n_runs=0,
            by_split={},
        )
    grouped: dict[int, list[float]] = defaultdict(list)
    for split_seed, _training_seed, value in finite:
        grouped[split_seed].append(value)
    by_split = {
        split_seed: SplitMetricSummary(
            mean=statistics.fmean(values),
            std_seed=_sample_std(values),
            n_seeds=len(values),
        )
        for split_seed, values in sorted(grouped.items())
    }
    split_means = [summary.mean for summary in by_split.values()]
    within_variances = [summary.std_seed**2 for summary in by_split.values()]
    all_values = [value for _, _, value in finite]
    seed_counts = {summary.n_seeds for summary in by_split.values()}
    return VarianceSummary(
        mean=statistics.fmean(split_means),
        std_split=_sample_std(split_means),
        std_seed=math.sqrt(statistics.fmean(within_variances)),
        std_total=_sample_std(all_values),
        n_splits=len(by_split),
        n_seeds_per_split=next(iter(seed_counts)) if len(seed_counts) == 1 else 0,
        n_runs=len(finite),
        by_split=by_split,
    )


def _global_metric_values(metrics) -> dict[str, float]:
    values = {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.macro_f1,
    }
    values.update(metrics.values)
    return values


def _partition_summary(
    runs: list[EvaluationResult], partition: str
) -> dict[str, VarianceSummary]:
    observations: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for run in runs:
        values = _global_metric_values(getattr(run, partition))
        for metric_name, value in values.items():
            observations[metric_name].append(
                (run.split_seed, run.random_seed, float(value))
            )
    return {
        metric_name: _aggregate_observations(values)
        for metric_name, values in sorted(observations.items())
    }


def _class_wise_summary(
    runs: list[EvaluationResult],
) -> dict[str, dict[str, dict[str, VarianceSummary]]] | None:
    if not runs or runs[0].test.per_class is None:
        return None
    result: dict[str, dict[str, dict[str, VarianceSummary]]] = {}
    for partition in ("train", "validation", "test"):
        observations: dict[
            str, dict[str, list[tuple[int, int, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        for run in runs:
            rows = getattr(run, partition).per_class
            if rows is None:
                continue
            for row in rows:
                for metric_name in ("precision", "recall", "f1", "support"):
                    observations[row.label][metric_name].append(
                        (
                            run.split_seed,
                            run.random_seed,
                            float(getattr(row, metric_name)),
                        )
                    )
        result[partition] = {
            label: {
                metric_name: _aggregate_observations(values)
                for metric_name, values in sorted(metrics.items())
            }
            for label, metrics in sorted(observations.items())
        }
    return result


def _summarize_hidden(hidden_dim: int, runs: list[EvaluationResult]) -> HiddenDimensionSummary:
    return HiddenDimensionSummary(
        hidden_dim=hidden_dim,
        train=_partition_summary(runs, "train"),
        validation=_partition_summary(runs, "validation"),
        test=_partition_summary(runs, "test"),
        class_wise=_class_wise_summary(runs),
    )


def _prepare_unified_data(
    benchmark,
    *,
    dataset: str,
    graph: str | None,
    graph_path: str | Path | None = None,
    graph_name: str | None = None,
    variant: str = "default",
    options: Iterable[str] | None = None,
    split_seeds: Iterable[int] | None = None,
    feature_seed: int = 42,
    feature_dim: int = 384,
    initial_embedding_model: str | None = None,
    embedding_batch_size: int = 256,
    embedding_device: str = "auto",
    embedding_cache: bool = True,
    embedding_progress: bool = True,
    embedding_cache_dir: str | Path | None = None,
    strict_coverage: bool = True,
    verify_graph: bool = False,
    no_graph: bool = False,
) -> _PreparedUnifiedData:
    manifest = benchmark._store.manifest(dataset)
    resolved_split_seeds = (
        tuple(sorted(manifest.splits))
        if split_seeds is None
        else _validate_unique_positive(split_seeds, "split_seeds")
    )
    missing_splits = set(resolved_split_seeds) - set(manifest.splits)
    if missing_splits:
        raise ValueError(
            f"Unknown split seeds {sorted(missing_splits)}. "
            f"Available: {sorted(manifest.splits)}"
        )
    for split_seed in resolved_split_seeds:
        benchmark._store.ensure_asset(manifest.splits[split_seed])
    first_split = resolved_split_seeds[0]
    if no_graph:
        base_bundle = benchmark.get_no_graph_data(
            dataset,
            split_seed=first_split,
            feature_dim=feature_dim,
            initial_embedding_model=initial_embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_device=embedding_device,
            embedding_cache=embedding_cache,
            embedding_progress=embedding_progress,
            embedding_cache_dir=embedding_cache_dir,
            feature_seed=feature_seed,
        )
    else:
        base_bundle = benchmark.get_data(
            dataset,
            graph,
            graph_path=graph_path,
            graph_name=graph_name,
            split_seed=first_split,
            variant=variant,
            options=options,
            feature_dim=feature_dim,
            initial_embedding_model=initial_embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_device=embedding_device,
            embedding_cache=embedding_cache,
            embedding_progress=embedding_progress,
            embedding_cache_dir=embedding_cache_dir,
            feature_seed=feature_seed,
            strict_coverage=strict_coverage,
            verify_graph=verify_graph,
        )
    bundles = {
        split_seed: (
            base_bundle
            if split_seed == first_split
            else replace_bundle_split(
                base_bundle,
                split_asset=manifest.splits[split_seed],
                split_seed=split_seed,
            )
        )
        for split_seed in resolved_split_seeds
    }
    return _PreparedUnifiedData(base_bundle, bundles, resolved_split_seeds)


def run_unified_evaluation(
    benchmark,
    *,
    dataset: str,
    graph: str | None,
    model: Any,
    graph_path: str | Path | None = None,
    graph_name: str | None = None,
    variant: str = "default",
    options: Iterable[str] | None = None,
    hidden_dims: Iterable[int] = DEFAULT_HIDDEN_DIMS,
    split_seeds: Iterable[int] | None = None,
    training_seeds: Iterable[int] = DEFAULT_TRAINING_SEEDS,
    feature_seed: int = 42,
    feature_dim: int = 384,
    initial_embedding_model: str | None = None,
    embedding_batch_size: int = 256,
    embedding_device: str = "auto",
    embedding_cache: bool = True,
    embedding_progress: bool = True,
    embedding_cache_dir: str | Path | None = None,
    metrics: str | Iterable[str] | None = "all",
    selection_metric: str = "f1_macro",
    class_wise: bool = True,
    track_training: bool = False,
    num_layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    batch_size: int = 512,
    num_neighbors: Iterable[int] = (200, 200),
    shuffle: bool = False,
    num_workers: int = 0,
    max_epochs: int = 100,
    patience: int = 100,
    min_delta: float = 0.0,
    device: str = "auto",
    strict_coverage: bool = True,
    verify_graph: bool = False,
    model_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = True,
    epoch_verbose: bool = False,
    log_every: int = 10,
    persist: bool = True,
    resume: bool = True,
    results_dir: str | Path | None = None,
    _prepared_data: _PreparedUnifiedData | None = None,
) -> UnifiedEvaluationResult:
    """Run the Cartesian product hidden × split × training seed.

    Hidden size is selected exclusively from the grand mean validation score.
    Test metrics never participate in model selection.
    """
    resolved_hidden_dims = _validate_unique_positive(hidden_dims, "hidden_dims")
    resolved_training_seeds = _validate_unique_positive(
        training_seeds, "training_seeds"
    )
    resolved_num_neighbors = tuple(num_neighbors)
    no_graph = isinstance(model, str) and model == "MLP"
    effective_dropout = (
        float(dict(model_kwargs or {}).get("dropout", 0.0))
        if no_graph
        else dropout
    )
    prepared_data = _prepared_data or _prepare_unified_data(
        benchmark,
        dataset=dataset,
        graph=graph,
        graph_path=graph_path,
        graph_name=graph_name,
        variant=variant,
        options=options,
        split_seeds=split_seeds,
        feature_seed=feature_seed,
        feature_dim=feature_dim,
        initial_embedding_model=initial_embedding_model,
        embedding_batch_size=embedding_batch_size,
        embedding_device=embedding_device,
        embedding_cache=embedding_cache,
        embedding_progress=embedding_progress,
        embedding_cache_dir=embedding_cache_dir,
        strict_coverage=strict_coverage,
        verify_graph=verify_graph,
        no_graph=no_graph,
    )
    resolved_split_seeds = prepared_data.split_seeds
    metric_names = resolve_metric_names(metrics)
    resolved_selection_metric = normalize_metric_name(selection_metric)
    if resolved_selection_metric not in metric_names:
        metric_names = (*metric_names, resolved_selection_metric)

    base_bundle = prepared_data.base_bundle
    bundles = prepared_data.bundles
    model_label = _model_label(model)
    resolved_results_root = (
        _resolve_results_root(benchmark, results_dir) if persist else None
    )
    protocol_base = {
        "schema_version": 1,
        "dataset": dataset,
        "graph": base_bundle.graph,
        "graph_identity": _graph_identity(base_bundle),
        "variant": base_bundle.variant,
        "model": model_label,
        "initial_embedding_model": base_bundle.initial_embedding_model,
        "feature_seed": feature_seed,
        "feature_dim": int(base_bundle.data.x.size(1)),
        "num_layers": num_layers,
        "dropout": effective_dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "training_mode": "full_batch" if no_graph else "neighbor_sampling",
        "batch_size": 0 if no_graph else batch_size,
        "num_neighbors": () if no_graph else resolved_num_neighbors,
        "shuffle": False if no_graph else shuffle,
        "num_workers": 0 if no_graph else num_workers,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "metrics": metric_names,
        "selection_metric": resolved_selection_metric,
        "class_wise": class_wise,
        "track_training": track_training,
        "model_kwargs": dict(model_kwargs or {}),
    }
    if verbose and resolved_results_root is not None:
        print(f"Checkpoints: {resolved_results_root}", flush=True)

    total_runs = (
        len(resolved_hidden_dims)
        * len(resolved_split_seeds)
        * len(resolved_training_seeds)
    )
    runs: list[EvaluationResult] = []
    run_number = 0
    resumed_runs = 0
    completed_runs = 0
    for hidden_dim in resolved_hidden_dims:
        for split_seed in resolved_split_seeds:
            for training_seed in resolved_training_seeds:
                run_number += 1
                run_protocol = {
                    **protocol_base,
                    "hidden_dim": hidden_dim,
                    "split_seed": split_seed,
                    "training_seed": training_seed,
                }
                protocol_fingerprint = _fingerprint(run_protocol)
                checkpoint_path = (
                    None
                    if resolved_results_root is None
                    else _run_checkpoint(
                        results_root=resolved_results_root,
                        dataset=dataset,
                        graph=base_bundle.graph,
                        initial_embedding_model=base_bundle.initial_embedding_model,
                        model=model_label,
                        hidden_dim=hidden_dim,
                        split_seed=split_seed,
                        training_seed=training_seed,
                        fingerprint=protocol_fingerprint,
                    )
                )
                prefix = (
                    f"[{run_number:03d}/{total_runs}] hidden={hidden_dim} | "
                    f"split={split_seed} | training_seed={training_seed}"
                )
                if resume and checkpoint_path is not None and checkpoint_path.is_file():
                    run_result = _load_checkpoint(
                        checkpoint_path, protocol_fingerprint
                    )
                    runs.append(run_result)
                    resumed_runs += 1
                    if verbose:
                        print(f"{prefix} | resumed", flush=True)
                    continue
                if verbose:
                    print(f"{prefix} | running", flush=True)
                run_result = evaluate_bundle(
                    bundles[split_seed],
                    copy.deepcopy(model),
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=effective_dropout,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    num_neighbors=resolved_num_neighbors,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    max_epochs=max_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    random_seed=training_seed,
                    device=device,
                    model_kwargs=model_kwargs,
                    metrics=metric_names,
                    selection_metric=resolved_selection_metric,
                    class_wise=class_wise,
                    track_training=track_training,
                    verbose=epoch_verbose,
                    log_every=log_every,
                )
                runs.append(run_result)
                completed_runs += 1
                if checkpoint_path is not None:
                    _atomic_json_write(
                        checkpoint_path,
                        {
                            "schema_version": 1,
                            "protocol_fingerprint": protocol_fingerprint,
                            "protocol": run_protocol,
                            "result": run_result.to_dict(),
                        },
                    )

    if verbose:
        print(
            f"Unified evaluation complete | loaded={resumed_runs} | "
            f"trained={completed_runs}",
            flush=True,
        )

    runs_per_hidden = len(resolved_split_seeds) * len(resolved_training_seeds)
    by_hidden_dim = {}
    for hidden_index, hidden_dim in enumerate(resolved_hidden_dims):
        start = hidden_index * runs_per_hidden
        hidden_runs = runs[start : start + runs_per_hidden]
        by_hidden_dim[hidden_dim] = _summarize_hidden(hidden_dim, hidden_runs)
    lower_is_better = resolved_selection_metric == "log_loss"
    selectable = {
        hidden: summary.validation[resolved_selection_metric].mean
        for hidden, summary in by_hidden_dim.items()
    }
    selected_hidden_dim = (
        min(selectable, key=lambda hidden: (selectable[hidden], hidden))
        if lower_is_better
        else max(selectable, key=lambda hidden: (selectable[hidden], -hidden))
    )
    model_name = runs[0].model
    grid_protocol = {
        **protocol_base,
        "hidden_dims": resolved_hidden_dims,
        "split_seeds": resolved_split_seeds,
        "training_seeds": resolved_training_seeds,
    }
    experiment_fingerprint = _fingerprint(grid_protocol)
    export_dir = (
        None
        if resolved_results_root is None
        else resolved_results_root
        / _slug(dataset)
        / _slug(base_bundle.graph)
        / _slug(base_bundle.initial_embedding_model or "random")
        / _slug(model_label)
        / f"experiment_{experiment_fingerprint[:16]}"
    )
    result = UnifiedEvaluationResult(
        dataset=dataset,
        graph=base_bundle.graph,
        model=model_name,
        initial_embedding_model=base_bundle.initial_embedding_model,
        hidden_dims=resolved_hidden_dims,
        split_seeds=resolved_split_seeds,
        training_seeds=resolved_training_seeds,
        feature_seed=feature_seed,
        metric_names=metric_names,
        selection_metric=resolved_selection_metric,
        selected_hidden_dim=selected_hidden_dim,
        results_dir=resolved_results_root,
        export_dir=export_dir,
        resumed_runs=resumed_runs,
        completed_runs=completed_runs,
        runs=tuple(runs),
        by_hidden_dim=by_hidden_dim,
    )
    if resolved_results_root is not None and export_dir is not None:
        summary_path = (
            resolved_results_root
            / _slug(dataset)
            / _slug(base_bundle.graph)
            / _slug(base_bundle.initial_embedding_model or "random")
            / _slug(model_label)
            / f"unified_summary-{_fingerprint(grid_protocol)[:16]}.json"
        )
        _atomic_json_write(summary_path, result.summary_dict())
        _atomic_json_write(export_dir / "protocol.json", grid_protocol)
        _atomic_json_write(export_dir / "summary.json", result.summary_dict())
        _atomic_csv_write(
            export_dir / "runs.csv",
            _run_rows(result),
            fallback_fields=(
                "dataset", "graph", "model", "hidden_dim", "split_seed",
                "training_seed", "best_epoch",
            ),
        )
        _atomic_csv_write(
            export_dir / "summary.csv",
            _summary_rows(result),
            fallback_fields=(
                "hidden_dim", "partition", "metric", "mean", "std_split",
                "std_seed", "std_total", "n_runs",
            ),
        )
        _atomic_csv_write(
            export_dir / "class_metrics.csv",
            _class_metric_rows(result),
            fallback_fields=(
                "hidden_dim", "split_seed", "training_seed", "partition",
                "class_label", "precision", "recall", "f1", "support",
            ),
        )
        _atomic_csv_write(
            export_dir / "training_history.csv",
            _training_history_rows(result),
            fallback_fields=(
                "hidden_dim", "split_seed", "training_seed", "epoch",
                "train_loss", "validation_loss", "validation_accuracy",
                "validation_f1_macro",
            ),
        )
        _atomic_json_write(
            export_dir / "confusion_matrices.json",
            _confusion_matrices(result),
        )
    return result


def _model_label(model: Any) -> str:
    if isinstance(model, str):
        return model
    return getattr(model, "__name__", model.__class__.__name__)


def run_unified_suite(
    benchmark,
    *,
    dataset: str,
    graph: str | None,
    models: Iterable[Any],
    **kwargs: Any,
) -> UnifiedEvaluationSuite:
    """Evaluate several architectures while preparing graph features once."""
    resolved_models = tuple(models)
    if not resolved_models:
        raise ValueError("models must not be empty")
    labels = tuple(_model_label(model) for model in resolved_models)
    if len(set(labels)) != len(labels):
        raise ValueError("models must have unique names")

    normalized_kwargs = dict(kwargs)
    for name in ("hidden_dims", "split_seeds", "training_seeds", "num_neighbors"):
        value = normalized_kwargs.get(name)
        if value is not None:
            normalized_kwargs[name] = tuple(value)
    metric_value = normalized_kwargs.get("metrics")
    if metric_value is not None and not isinstance(metric_value, str):
        normalized_kwargs["metrics"] = tuple(metric_value)

    preparation_names = {
        "graph_path",
        "graph_name",
        "variant",
        "options",
        "split_seeds",
        "feature_seed",
        "feature_dim",
        "initial_embedding_model",
        "embedding_batch_size",
        "embedding_device",
        "embedding_cache",
        "embedding_progress",
        "embedding_cache_dir",
        "strict_coverage",
        "verify_graph",
    }
    preparation_kwargs = {
        name: value
        for name, value in normalized_kwargs.items()
        if name in preparation_names
    }
    graph_prepared = (
        _prepare_unified_data(
            benchmark,
            dataset=dataset,
            graph=graph,
            **preparation_kwargs,
        )
        if any(label != "MLP" for label in labels)
        else None
    )
    no_graph_prepared = (
        _prepare_unified_data(
            benchmark,
            dataset=dataset,
            graph=None,
            no_graph=True,
            **preparation_kwargs,
        )
        if "MLP" in labels
        else None
    )

    results: dict[str, UnifiedEvaluationResult] = {}
    for index, (label, model) in enumerate(zip(labels, resolved_models), start=1):
        if normalized_kwargs.get("verbose", True):
            print(
                f"\n=== Model {index}/{len(resolved_models)}: {label} ===",
                flush=True,
            )
        results[label] = run_unified_evaluation(
            benchmark,
            dataset=dataset,
            graph=None if label == "MLP" else graph,
            model=model,
            _prepared_data=(
                no_graph_prepared if label == "MLP" else graph_prepared
            ),
            **normalized_kwargs,
        )

    selection_metric = next(iter(results.values())).selection_metric
    lower_is_better = selection_metric == "log_loss"
    validation_scores = {
        label: result.selected_summary.validation[selection_metric].mean
        for label, result in results.items()
    }
    selected_model = (
        min(validation_scores, key=lambda label: validation_scores[label])
        if lower_is_better
        else max(validation_scores, key=lambda label: validation_scores[label])
    )
    reference_label = next(
        (label for label in labels if label != "MLP"), labels[0]
    )
    first = results[reference_label]
    suite = UnifiedEvaluationSuite(
        dataset=dataset,
        graph=first.graph,
        models=labels,
        initial_embedding_model=first.initial_embedding_model,
        selection_metric=selection_metric,
        selected_model=selected_model,
        results_dir=first.results_dir,
        resumed_runs=sum(result.resumed_runs for result in results.values()),
        completed_runs=sum(result.completed_runs for result in results.values()),
        results=results,
    )
    if suite.results_dir is not None:
        suite_protocol = {
            "schema_version": 1,
            "dataset": dataset,
            "graph": first.graph,
            "models": labels,
            "initial_embedding_model": first.initial_embedding_model,
            "selection_metric": selection_metric,
            "model_grids": {
                label: {
                    "hidden_dims": result.hidden_dims,
                    "split_seeds": result.split_seeds,
                    "training_seeds": result.training_seeds,
                    "feature_seed": result.feature_seed,
                    "metric_names": result.metric_names,
                }
                for label, result in results.items()
            },
        }
        summary_path = (
            suite.results_dir
            / _slug(dataset)
            / _slug(first.graph)
            / _slug(first.initial_embedding_model or "random")
            / f"unified_suite_summary-{_fingerprint(suite_protocol)[:16]}.json"
        )
        _atomic_json_write(summary_path, suite.summary_dict())
    return suite
