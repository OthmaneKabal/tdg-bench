"""Reproducible neighbor-sampled mini-batch node classification."""

from __future__ import annotations

import inspect
import math
import random
import statistics
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    import torch_geometric.typing
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import MessagePassing
    from torch_geometric.loader import NeighborLoader
except ImportError as exc:  # pragma: no cover - broken installation guard
    from tdg_bench.exceptions import OptionalDependencyError

    raise OptionalDependencyError(
        "Evaluation requires PyTorch and PyTorch Geometric. "
        "Reinstall them with: pip install --upgrade tdg-bench"
    ) from exc

from tdg_bench.data_preparation.builder import DataBundle
from tdg_bench.exceptions import OptionalDependencyError
from tdg_bench.models import build_model
from tdg_bench.models.encoders import NodeClassifier
from tdg_bench.evaluation.metrics import (
    ClassMetrics,
    calculate_metrics,
    normalize_metric_name,
    resolve_metric_names,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    support: int
    values: dict[str, float]
    per_class: tuple[ClassMetrics, ...] | None
    confusion_matrix: tuple[tuple[int, ...], ...] | None


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    selection_metric: str
    selection_score: float
    train_accuracy: float | None = None
    train_macro_f1: float | None = None
    train_values: dict[str, float] | None = None
    validation_values: dict[str, float] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    dataset: str
    graph: str
    variant: str
    model: str
    initial_embedding_model: str | None
    split_seed: int
    random_seed: int
    device: str
    best_epoch: int
    epochs_ran: int
    metric_names: tuple[str, ...]
    selection_metric: str
    class_wise: bool
    training_mode: str
    batch_size: int
    num_neighbors: tuple[int, ...]
    shuffle: bool
    track_training: bool
    train: ClassificationMetrics
    validation: ClassificationMetrics
    test: ClassificationMetrics
    history: tuple[EpochMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    runs: tuple[EvaluationResult, ...]
    aggregate: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": self.aggregate,
        }


def _classification_metrics_from_dict(value: Mapping[str, Any]) -> ClassificationMetrics:
    raw_per_class = value.get("per_class")
    per_class = (
        None
        if raw_per_class is None
        else tuple(ClassMetrics(**row) for row in raw_per_class)
    )
    raw_matrix = value.get("confusion_matrix")
    matrix = (
        None
        if raw_matrix is None
        else tuple(tuple(int(item) for item in row) for row in raw_matrix)
    )
    return ClassificationMetrics(
        loss=float(value["loss"]),
        accuracy=float(value["accuracy"]),
        macro_f1=float(value["macro_f1"]),
        support=int(value["support"]),
        values={name: float(score) for name, score in value["values"].items()},
        per_class=per_class,
        confusion_matrix=matrix,
    )


def evaluation_result_from_dict(value: Mapping[str, Any]) -> EvaluationResult:
    """Reconstruct one run from an atomic unified-evaluation checkpoint."""
    return EvaluationResult(
        dataset=str(value["dataset"]),
        graph=str(value["graph"]),
        variant=str(value["variant"]),
        model=str(value["model"]),
        initial_embedding_model=value.get("initial_embedding_model"),
        split_seed=int(value["split_seed"]),
        random_seed=int(value["random_seed"]),
        device=str(value["device"]),
        best_epoch=int(value["best_epoch"]),
        epochs_ran=int(value["epochs_ran"]),
        metric_names=tuple(value["metric_names"]),
        selection_metric=str(value["selection_metric"]),
        class_wise=bool(value["class_wise"]),
        training_mode=str(value.get("training_mode", "full_batch")),
        batch_size=int(value.get("batch_size", 0)),
        num_neighbors=tuple(int(item) for item in value.get("num_neighbors", ())),
        shuffle=bool(value.get("shuffle", False)),
        track_training=bool(value.get("track_training", bool(value.get("history")))),
        train=_classification_metrics_from_dict(value["train"]),
        validation=_classification_metrics_from_dict(value["validation"]),
        test=_classification_metrics_from_dict(value["test"]),
        history=tuple(EpochMetrics(**row) for row in value["history"]),
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def _call_factory(factory: Callable[..., nn.Module], kwargs: dict[str, Any]) -> nn.Module:
    """Pass only declared keywords unless the custom factory accepts **kwargs."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    accepts_all = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = kwargs if accepts_all else {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return factory(**selected)


def _prepare_model(
    model: str | nn.Module | Callable[..., nn.Module],
    bundle: DataBundle,
    *,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    model_kwargs: Mapping[str, Any] | None,
) -> tuple[nn.Module, str]:
    common = {
        "input_dim": int(bundle.data.x.size(1)),
        "num_classes": int(bundle.data.num_classes),
        "num_relations": int(bundle.data.num_relations),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        **dict(model_kwargs or {}),
    }
    if isinstance(model, str):
        if model == "MLP":
            mlp_options = dict(model_kwargs or {})
            mlp_dropout = mlp_options.pop("dropout", 0.0)
            mlp_arguments = {
                "input_dim": common["input_dim"],
                "hidden_dim": hidden_dim,
                "num_classes": common["num_classes"],
                "dropout": mlp_dropout,
                **mlp_options,
            }
            return build_model(model, **mlp_arguments), model
        encoder_arguments = dict(common)
        encoder_arguments.pop("num_classes")
        if model != "RGCN":
            encoder_arguments.pop("num_relations")
        encoder = build_model(model, **encoder_arguments)
        return NodeClassifier(encoder, common["num_classes"]), model
    from tdg_bench.models.pyg_adapter import (
        PyGConvolutionClassifier,
        is_pyg_convolution_type,
    )

    if is_pyg_convolution_type(model):
        adapter_kwargs = dict(model_kwargs or {})
        unknown = set(adapter_kwargs) - {"conv_kwargs", "activation"}
        if unknown:
            raise ValueError(
                "PyG convolution model_kwargs accepts only conv_kwargs and "
                f"activation; unknown keys: {sorted(unknown)}"
            )
        network = PyGConvolutionClassifier(
            model,
            input_dim=common["input_dim"],
            hidden_dim=hidden_dim,
            num_classes=common["num_classes"],
            num_relations=common["num_relations"],
            num_layers=num_layers,
            dropout=dropout,
            **adapter_kwargs,
        )
        return network, model.__name__
    if isinstance(model, MessagePassing):
        raise TypeError(
            "Pass the PyG convolution class, not an initialized instance: "
            f"model={model.__class__.__name__}"
        )
    if isinstance(model, nn.Module):
        return model, model.__class__.__name__
    if callable(model):
        built = _call_factory(model, common)
        if not isinstance(built, nn.Module):
            raise TypeError("A custom model factory must return torch.nn.Module")
        return built, getattr(model, "__name__", built.__class__.__name__)
    raise TypeError("model must be a built-in name, torch.nn.Module, or model factory")


def _forward(model: nn.Module, data, *, num_classes: int) -> torch.Tensor:
    logits = model(data.x, data.edge_index, data.edge_type, data.relation_x)
    expected = (data.num_nodes, num_classes)
    if tuple(logits.shape) != expected:
        raise ValueError(
            f"Model output must have shape {expected}, got {tuple(logits.shape)}"
        )
    return logits


def _resolve_num_neighbors(
    num_neighbors: Iterable[int], *, num_layers: int
) -> tuple[int, ...]:
    resolved = tuple(num_neighbors)
    if len(resolved) != num_layers:
        raise ValueError(
            f"num_neighbors must contain one value per GNN layer "
            f"({num_layers}), got {len(resolved)}"
        )
    if any(
        not isinstance(value, int) or value < -1
        for value in resolved
    ):
        raise ValueError("num_neighbors values must be integers >= 0 or -1")
    return resolved


def _build_loaders(
    data,
    *,
    num_neighbors: tuple[int, ...],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    if not (
        torch_geometric.typing.WITH_PYG_LIB
        or torch_geometric.typing.WITH_TORCH_SPARSE
    ):
        raise OptionalDependencyError(
            "Mini-batch neighbor sampling requires either 'pyg-lib' or "
            "PyTorch-compatible installations of both 'torch-scatter' and "
            "'torch-sparse'. Restart the Python kernel after installing them."
        )
    common = {
        "num_neighbors": list(num_neighbors),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": NeighborLoader(
            data,
            input_nodes=data.train_mask,
            shuffle=shuffle,
            **common,
        ),
        "validation": NeighborLoader(
            data,
            input_nodes=data.val_mask,
            shuffle=False,
            **common,
        ),
        "test": NeighborLoader(
            data,
            input_nodes=data.test_mask,
            shuffle=False,
            **common,
        ),
    }


def _seed_partition(logits, batch):
    seed_count = int(batch.batch_size)
    seed_logits = logits[:seed_count]
    seed_labels = batch.y[:seed_count]
    valid = seed_labels >= 0
    return seed_logits[valid], seed_labels[valid]


def _train_epoch(
    network: nn.Module,
    loader,
    optimizer,
    *,
    device: torch.device,
    num_classes: int,
) -> float:
    network.train()
    total_loss = 0.0
    total_support = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(network, batch, num_classes=num_classes)
        seed_logits, seed_labels = _seed_partition(logits, batch)
        if seed_labels.numel() == 0:
            continue
        loss = F.cross_entropy(seed_logits, seed_labels)
        loss.backward()
        optimizer.step()
        support = int(seed_labels.numel())
        total_loss += float(loss.detach()) * support
        total_support += support
    if total_support == 0:
        raise ValueError("Training loader contains no labelled input nodes")
    return total_loss / total_support


def _evaluate_loader(
    network: nn.Module,
    loader,
    *,
    device: torch.device,
    num_classes: int,
    metric_names: tuple[str, ...],
    label_names: tuple[str, ...],
    class_wise: bool,
) -> ClassificationMetrics:
    network.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = _forward(network, batch, num_classes=num_classes)
            seed_logits, seed_labels = _seed_partition(logits, batch)
            if seed_labels.numel() == 0:
                continue
            logits_parts.append(seed_logits.detach().cpu())
            label_parts.append(seed_labels.detach().cpu())
    if not label_parts:
        raise ValueError("Evaluation loader contains no labelled input nodes")
    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    mask = torch.ones(labels.size(0), dtype=torch.bool)
    return _metrics(
        logits,
        labels,
        mask,
        metric_names=metric_names,
        label_names=label_names,
        class_wise=class_wise,
    )


def _metrics(
    logits,
    labels,
    mask,
    *,
    metric_names: tuple[str, ...],
    label_names: tuple[str, ...],
    class_wise: bool,
) -> ClassificationMetrics:
    required = tuple(dict.fromkeys((*metric_names, "accuracy", "f1_macro")))
    loss, support, computed, per_class, matrix = calculate_metrics(
        logits=logits,
        labels=labels,
        mask=mask,
        metric_names=required,
        label_names=label_names,
        class_wise=class_wise,
    )
    return ClassificationMetrics(
        loss=loss,
        accuracy=computed["accuracy"],
        macro_f1=computed["f1_macro"],
        support=support,
        values={name: computed[name] for name in metric_names},
        per_class=per_class,
        confusion_matrix=matrix,
    )


def _evaluate_no_graph_mlp(
    bundle: DataBundle,
    *,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    random_seed: int,
    device: str | torch.device,
    model_kwargs: Mapping[str, Any] | None,
    metric_names: tuple[str, ...],
    selection_metric: str,
    class_wise: bool,
    track_training: bool,
    verbose: bool,
    log_every: int,
) -> EvaluationResult:
    """Train the historical feature-only MLP with stratified mini-batches."""
    _set_random_seed(random_seed)
    resolved_device = _resolve_device(device)
    network, model_name = _prepare_model(
        "MLP",
        bundle,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=dropout,
        model_kwargs=model_kwargs,
    )
    network = network.to(resolved_device)
    data = bundle.data.clone().to(resolved_device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    label_names = bundle.label_names

    def evaluate_mask(mask, names: tuple[str, ...]) -> ClassificationMetrics:
        network.eval()
        indices = mask.nonzero(as_tuple=False).view(-1)
        logits_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, indices.numel(), batch_size):
                batch_indices = indices[start : start + batch_size]
                logits_parts.append(network(data.x[batch_indices]))
                label_parts.append(data.y[batch_indices])
        logits = torch.cat(logits_parts, dim=0)
        labels = torch.cat(label_parts, dim=0)
        return _metrics(
            logits,
            labels,
            torch.ones(labels.size(0), dtype=torch.bool, device=labels.device),
            metric_names=names,
            label_names=label_names,
            class_wise=class_wise,
        )

    def stratified_batches(epoch: int) -> list[torch.Tensor]:
        train_indices = data.train_mask.nonzero(as_tuple=False).view(-1).cpu()
        train_labels = data.y[train_indices.to(data.y.device)].detach().cpu()
        generator = torch.Generator(device="cpu").manual_seed(random_seed + epoch - 1)
        queues: dict[int, torch.Tensor] = {}
        offsets: dict[int, int] = {}
        for class_id in sorted(int(value) for value in train_labels.unique().tolist()):
            class_indices = train_indices[train_labels == class_id]
            queues[class_id] = class_indices[
                torch.randperm(class_indices.numel(), generator=generator)
            ]
            offsets[class_id] = 0

        batches: list[torch.Tensor] = []
        remaining_total = int(train_indices.numel())
        while remaining_total:
            current_size = min(batch_size, remaining_total)
            remaining = {
                class_id: int(indices.numel()) - offsets[class_id]
                for class_id, indices in queues.items()
            }
            ideals = {
                class_id: current_size * count / remaining_total
                for class_id, count in remaining.items()
            }
            allocations = {
                class_id: min(count, math.floor(ideals[class_id]))
                for class_id, count in remaining.items()
            }
            missing = current_size - sum(allocations.values())
            while missing:
                candidates = [
                    class_id
                    for class_id, count in remaining.items()
                    if allocations[class_id] < count
                ]
                class_id = max(
                    candidates,
                    key=lambda value: (ideals[value] - allocations[value], -value),
                )
                allocations[class_id] += 1
                missing -= 1

            parts = []
            for class_id, count in allocations.items():
                start = offsets[class_id]
                parts.append(queues[class_id][start : start + count])
                offsets[class_id] += count
            batch = torch.cat(parts)
            batch = batch[torch.randperm(batch.numel(), generator=generator)]
            batches.append(batch.to(resolved_device))
            remaining_total -= current_size
        return batches

    lower_is_better = selection_metric == "log_loss"
    best_score = math.inf if lower_is_better else -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[EpochMetrics] = []
    epochs_ran = 0
    if verbose:
        print(
            f"Training MLP | dataset={bundle.dataset} | graph=none | "
            f"split={bundle.split_seed} | device={resolved_device} | "
            f"features={getattr(bundle, 'initial_embedding_model', None) or 'random'} | "
            f"batch_size={batch_size} | stratified=True",
            flush=True,
        )
    for epoch in range(1, max_epochs + 1):
        network.train()
        epoch_loss = 0.0
        trained_nodes = 0
        for batch_indices in stratified_batches(epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = network(data.x[batch_indices])
            loss = F.cross_entropy(logits, data.y[batch_indices])
            loss.backward()
            optimizer.step()
            batch_nodes = int(batch_indices.numel())
            epoch_loss += float(loss.detach()) * batch_nodes
            trained_nodes += batch_nodes
        train_loss = epoch_loss / trained_nodes
        train_at_epoch = (
            evaluate_mask(data.train_mask, metric_names) if track_training else None
        )
        validation = evaluate_mask(
            data.val_mask,
            metric_names if track_training else (selection_metric,),
        )
        selection_score = validation.values[selection_metric]
        if math.isnan(selection_score):
            raise ValueError(
                f"selection_metric={selection_metric!r} is undefined on the validation split"
            )
        epochs_ran = epoch
        if track_training:
            history.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation.loss,
                    validation_accuracy=validation.accuracy,
                    validation_macro_f1=validation.macro_f1,
                    selection_metric=selection_metric,
                    selection_score=selection_score,
                    train_accuracy=train_at_epoch.accuracy,
                    train_macro_f1=train_at_epoch.macro_f1,
                    train_values=dict(train_at_epoch.values),
                    validation_values=dict(validation.values),
                )
            )
        improved = (
            selection_score < best_score - min_delta
            if lower_is_better
            else selection_score > best_score + min_delta
        )
        if improved:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        stopping = stale_epochs >= patience
        if verbose and (epoch == 1 or epoch % log_every == 0 or stopping):
            if train_at_epoch is None:
                train_at_epoch = evaluate_mask(data.train_mask, (selection_metric,))
            print(
                f"epoch={epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_loss={validation.loss:.4f} | "
                f"train_{selection_metric}="
                f"{train_at_epoch.values[selection_metric]:.4f} | "
                f"val_{selection_metric}={selection_score:.4f} | "
                f"stale={stale_epochs}/{patience}",
                flush=True,
            )
        if stopping:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    network.load_state_dict(best_state)
    train_metrics = evaluate_mask(data.train_mask, metric_names)
    validation_metrics = evaluate_mask(data.val_mask, metric_names)
    test_metrics = evaluate_mask(data.test_mask, metric_names)
    return EvaluationResult(
        dataset=bundle.dataset,
        graph="no_graph",
        variant="none",
        model=model_name,
        initial_embedding_model=getattr(bundle, "initial_embedding_model", None),
        split_seed=bundle.split_seed,
        random_seed=random_seed,
        device=str(resolved_device),
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        metric_names=metric_names,
        selection_metric=selection_metric,
        class_wise=class_wise,
        training_mode="stratified_minibatch",
        batch_size=batch_size,
        num_neighbors=(),
        shuffle=True,
        track_training=track_training,
        train=train_metrics,
        validation=validation_metrics,
        test=test_metrics,
        history=tuple(history),
    )


def evaluate_bundle(
    bundle: DataBundle,
    model: str | nn.Module | Callable[..., nn.Module] = "GCN",
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    batch_size: int = 512,
    num_neighbors: Iterable[int] = (200, 200),
    shuffle: bool = False,
    num_workers: int = 0,
    max_epochs: int = 200,
    patience: int = 30,
    min_delta: float = 0.0,
    random_seed: int = 42,
    device: str | torch.device = "auto",
    model_kwargs: Mapping[str, Any] | None = None,
    metrics: str | Iterable[str] | None = None,
    selection_metric: str = "f1_macro",
    class_wise: bool = False,
    track_training: bool = False,
    verbose: bool = False,
    log_every: int = 10,
) -> EvaluationResult:
    """Train once and report metrics at the best validation macro-F1 epoch.

    A custom module must return logits for every node in each sampled batch,
    with shape ``[sampled_nodes, num_classes]``. A custom factory receives the compatible
    subset of ``input_dim``, ``num_classes``, ``num_relations``,
    ``hidden_dim``, ``num_layers`` and ``dropout``.
    """
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    no_graph_mlp = isinstance(model, str) and model == "MLP"
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    metric_names = resolve_metric_names(metrics)
    selection_metric = normalize_metric_name(selection_metric)

    if no_graph_mlp:
        return _evaluate_no_graph_mlp(
            bundle,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            random_seed=random_seed,
            device=device,
            model_kwargs=model_kwargs,
            metric_names=metric_names,
            selection_metric=selection_metric,
            class_wise=class_wise,
            track_training=track_training,
            verbose=verbose,
            log_every=log_every,
        )

    resolved_num_neighbors = _resolve_num_neighbors(
        num_neighbors, num_layers=num_layers
    )

    _set_random_seed(random_seed)
    resolved_device = _resolve_device(device)
    network, model_name = _prepare_model(
        model,
        bundle,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        model_kwargs=model_kwargs,
    )
    network = network.to(resolved_device)
    data = bundle.data.clone().cpu()
    loaders = _build_loaders(
        data,
        num_neighbors=resolved_num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    optimizer = torch.optim.Adam(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    lower_is_better = selection_metric == "log_loss"
    best_score = math.inf if lower_is_better else -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_ran = 0
    history: list[EpochMetrics] = []
    initial_embedding_model = getattr(bundle, "initial_embedding_model", None)
    label_names = getattr(
        bundle,
        "label_names",
        tuple(str(index) for index in range(data.num_classes)),
    )
    if verbose:
        print(
            f"Training {model_name} | graph={bundle.dataset}/{bundle.graph} | "
            f"split={bundle.split_seed} | device={resolved_device} | "
            f"features={initial_embedding_model or 'random'} | "
            f"batch_size={batch_size} | neighbors={list(resolved_num_neighbors)}",
            flush=True,
        )
    for epoch in range(1, max_epochs + 1):
        epoch_train_loss = _train_epoch(
            network,
            loaders["train"],
            optimizer,
            device=resolved_device,
            num_classes=int(data.num_classes),
        )
        train_at_epoch = (
            _evaluate_loader(
                network,
                loaders["train"],
                device=resolved_device,
                num_classes=int(data.num_classes),
                metric_names=metric_names,
                label_names=label_names,
                class_wise=False,
            )
            if track_training
            else None
        )
        validation = _evaluate_loader(
            network,
            loaders["validation"],
            device=resolved_device,
            num_classes=int(data.num_classes),
            metric_names=metric_names if track_training else (selection_metric,),
            label_names=label_names,
            class_wise=False,
        )
        selection_score = validation.values[selection_metric]
        if math.isnan(selection_score):
            raise ValueError(
                f"selection_metric={selection_metric!r} is undefined on the validation split"
            )
        epochs_ran = epoch
        if track_training:
            history.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=epoch_train_loss,
                    validation_loss=validation.loss,
                    validation_accuracy=validation.accuracy,
                    validation_macro_f1=validation.macro_f1,
                    selection_metric=selection_metric,
                    selection_score=selection_score,
                    train_accuracy=train_at_epoch.accuracy,
                    train_macro_f1=train_at_epoch.macro_f1,
                    train_values=dict(train_at_epoch.values),
                    validation_values=dict(validation.values),
                )
            )
        improved = (
            selection_score < best_score - min_delta
            if lower_is_better
            else selection_score > best_score + min_delta
        )
        if improved:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        stopping = stale_epochs >= patience
        if verbose and (epoch == 1 or epoch % log_every == 0 or stopping):
            if train_at_epoch is None:
                train_at_epoch = _evaluate_loader(
                    network,
                    loaders["train"],
                    device=resolved_device,
                    num_classes=int(data.num_classes),
                    metric_names=(selection_metric,),
                    label_names=label_names,
                    class_wise=False,
                )
            marker = " *" if improved else ""
            print(
                f"epoch={epoch:03d} | train_loss={epoch_train_loss:.4f} | "
                f"val_loss={validation.loss:.4f} | "
                f"train_{selection_metric}="
                f"{train_at_epoch.values[selection_metric]:.4f} | "
                f"val_{selection_metric}={selection_score:.4f} | "
                f"stale={stale_epochs}/{patience}{marker}",
                flush=True,
            )
        if stopping:
            break

    if best_state is None:  # defensive: max_epochs >= 1 always creates a state
        raise RuntimeError("Training did not produce a model state")
    network.load_state_dict(best_state)
    train_metrics = _evaluate_loader(
        network,
        loaders["train"],
        device=resolved_device,
        num_classes=int(data.num_classes),
        metric_names=metric_names,
        label_names=label_names,
        class_wise=class_wise,
    )
    validation_metrics = _evaluate_loader(
        network,
        loaders["validation"],
        device=resolved_device,
        num_classes=int(data.num_classes),
        metric_names=metric_names,
        label_names=label_names,
        class_wise=class_wise,
    )
    test_metrics = _evaluate_loader(
        network,
        loaders["test"],
        device=resolved_device,
        num_classes=int(data.num_classes),
        metric_names=metric_names,
        label_names=label_names,
        class_wise=class_wise,
    )

    if verbose:
        print(
            f"best_epoch={best_epoch} | test_acc={test_metrics.accuracy:.4f} | "
            f"test_macro_f1={test_metrics.macro_f1:.4f}",
            flush=True,
        )

    return EvaluationResult(
        dataset=bundle.dataset,
        graph=bundle.graph,
        variant=bundle.variant,
        model=model_name,
        initial_embedding_model=initial_embedding_model,
        split_seed=bundle.split_seed,
        random_seed=random_seed,
        device=str(resolved_device),
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        metric_names=metric_names,
        selection_metric=selection_metric,
        class_wise=class_wise,
        training_mode="neighbor_sampling",
        batch_size=batch_size,
        num_neighbors=resolved_num_neighbors,
        shuffle=shuffle,
        track_training=track_training,
        train=train_metrics,
        validation=validation_metrics,
        test=test_metrics,
        history=tuple(history),
    )


def aggregate_results(results: Iterable[EvaluationResult]) -> BenchmarkResult:
    runs = tuple(results)
    if not runs:
        raise ValueError("At least one evaluation result is required")
    aggregate: dict[str, dict[str, float]] = {}
    for partition in ("train", "validation", "test"):
        partition_metrics = [getattr(run, partition) for run in runs]
        values: dict[str, float] = {}
        for metric_name in ("loss", "accuracy", "macro_f1"):
            samples = [getattr(metrics, metric_name) for metrics in partition_metrics]
            values[f"{metric_name}_mean"] = statistics.fmean(samples)
            values[f"{metric_name}_std"] = statistics.pstdev(samples)
        common_names = set.intersection(
            *(set(metrics.values) for metrics in partition_metrics)
        )
        for metric_name in sorted(common_names):
            samples = [metrics.values[metric_name] for metrics in partition_metrics]
            values[f"{metric_name}_mean"] = statistics.fmean(samples)
            values[f"{metric_name}_std"] = statistics.pstdev(samples)
        aggregate[partition] = values
    return BenchmarkResult(runs=runs, aggregate=aggregate)

