"""Build deterministic PyTorch Geometric data from a TDGBench dataset."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tdg_bench.datasets.manifest import Asset, GraphArtifact
from tdg_bench.exceptions import DataValidationError, OptionalDependencyError
from tdg_bench.graphs.io import iter_graph_triples

try:
    import torch
    from openpyxl import load_workbook
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - exercised in minimal installations
    raise OptionalDependencyError(
        "get_data() requires the GNN dependencies. Install them with: "
        "pip install 'tdg-bench[gnn]'"
    ) from exc


@dataclass(frozen=True)
class GoldStandard:
    """Gold Standard rows in the exact order used by split indices."""

    terms: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class DataBundle:
    """One graph, Gold Standard and predefined split prepared for GNN use."""

    dataset: str
    graph: str
    variant: str
    split_seed: int
    artifact: GraphArtifact
    data: Any
    node_to_index: dict[str, int]
    index_to_node: tuple[str, ...]
    relation_to_index: dict[str, int]
    index_to_relation: tuple[str, ...]
    label_names: tuple[str, ...]
    gold_standard: GoldStandard
    gold_to_node: tuple[int | None, ...]
    missing_gold_terms: tuple[str, ...]
    initial_embedding_model: str | None = None
    embedding_cache_path: Path | None = None

    @property
    def coverage(self) -> float:
        if not self.gold_to_node:
            return 0.0
        found = sum(index is not None for index in self.gold_to_node)
        return found / len(self.gold_to_node)

    @property
    def split_sizes(self) -> dict[str, int]:
        return {
            "train": int(self.data.train_mask.sum()),
            "validation": int(self.data.val_mask.sum()),
            "test": int(self.data.test_mask.sum()),
        }


def _load_gold_standard(path: Path) -> GoldStandard:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise DataValidationError(f"Gold Standard is empty: {path}") from exc
        columns = {str(value).strip(): index for index, value in enumerate(header) if value is not None}
        missing = {"term", "label"} - set(columns)
        if missing:
            raise DataValidationError(
                f"Gold Standard {path} is missing columns: {sorted(missing)}"
            )
        terms: list[str] = []
        labels: list[str] = []
        for row_number, row in enumerate(rows, start=2):
            term_value = row[columns["term"]]
            label_value = row[columns["label"]]
            if term_value is None and label_value is None:
                continue
            if not isinstance(term_value, str) or not term_value.strip():
                raise DataValidationError(f"Invalid term at {path}:{row_number}")
            if not isinstance(label_value, str) or not label_value.strip():
                raise DataValidationError(f"Invalid label at {path}:{row_number}")
            terms.append(term_value.strip())
            labels.append(label_value.strip())
    finally:
        workbook.close()
    if not terms:
        raise DataValidationError(f"Gold Standard has no labelled terms: {path}")
    return GoldStandard(tuple(terms), tuple(labels))


def _scan_graph(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    nodes: set[str] = set()
    relations: set[str] = set()
    edge_count = 0
    for subject, predicate, obj in iter_graph_triples(path):
        nodes.add(subject)
        nodes.add(obj)
        relations.add(predicate)
        edge_count += 1
    if edge_count == 0:
        raise DataValidationError(f"Graph contains no triples: {path}")
    return tuple(sorted(nodes)), tuple(sorted(relations)), edge_count


def _build_edges(
    path: Path,
    node_to_index: dict[str, int],
    relation_to_index: dict[str, int],
    edge_count: int,
    *,
    chunk_size: int = 100_000,
):
    edge_index = torch.empty((2, edge_count), dtype=torch.long)
    edge_type = torch.empty(edge_count, dtype=torch.long)
    sources: list[int] = []
    targets: list[int] = []
    types: list[int] = []
    offset = 0

    def flush() -> None:
        nonlocal offset
        if not sources:
            return
        count = len(sources)
        edge_index[0, offset : offset + count] = torch.tensor(sources, dtype=torch.long)
        edge_index[1, offset : offset + count] = torch.tensor(targets, dtype=torch.long)
        edge_type[offset : offset + count] = torch.tensor(types, dtype=torch.long)
        offset += count
        sources.clear()
        targets.clear()
        types.clear()

    for subject, predicate, obj in iter_graph_triples(path):
        sources.append(node_to_index[subject])
        targets.append(node_to_index[obj])
        types.append(relation_to_index[predicate])
        if len(sources) >= chunk_size:
            flush()
    flush()
    if offset != edge_count:
        raise DataValidationError(
            f"Graph changed while being read: expected {edge_count} edges, got {offset}"
        )
    return edge_index, edge_type


def _load_split(asset: Asset, gold_size: int) -> dict[str, list[int]]:
    try:
        raw = json.loads(asset.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Could not read split {asset.path}: {exc}") from exc
    required = ("train_idx", "val_idx", "test_idx")
    if not isinstance(raw, dict) or any(name not in raw for name in required):
        raise DataValidationError(f"Split {asset.path} must contain {list(required)}")
    result: dict[str, list[int]] = {}
    seen_gold_indices: set[int] = set()
    for name in required:
        values = raw[name]
        if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
            raise DataValidationError(f"{asset.path}: {name} must contain integer indices")
        if len(values) != len(set(values)):
            raise DataValidationError(f"{asset.path}: {name} contains duplicate indices")
        if any(value < 0 or value >= gold_size for value in values):
            raise DataValidationError(f"{asset.path}: {name} contains an out-of-range index")
        overlap = seen_gold_indices.intersection(values)
        if overlap:
            raise DataValidationError(
                f"{asset.path}: Gold Standard index {min(overlap)} occurs in multiple partitions"
            )
        seen_gold_indices.update(values)
        result[name] = values
    return result


def _build_split_masks(
    *,
    split_asset: Asset,
    gold: GoldStandard,
    gold_to_node: tuple[int | None, ...],
    num_nodes: int,
) -> dict[str, Any]:
    split = _load_split(split_asset, len(gold.terms))
    masks = {
        "train_idx": torch.zeros(num_nodes, dtype=torch.bool),
        "val_idx": torch.zeros(num_nodes, dtype=torch.bool),
        "test_idx": torch.zeros(num_nodes, dtype=torch.bool),
    }
    assigned_nodes: dict[int, str] = {}
    for partition, gold_indices in split.items():
        for gold_index in gold_indices:
            node_index = gold_to_node[gold_index]
            if node_index is None:
                continue
            previous = assigned_nodes.get(node_index)
            if previous is not None and previous != partition:
                term = gold.terms[gold_index]
                raise DataValidationError(
                    f"Gold Standard term {term!r} maps to the same graph node in both "
                    f"{previous} and {partition}; fix duplicate terms or split leakage"
                )
            assigned_nodes[node_index] = partition
            masks[partition][node_index] = True
    return masks


def build_data_bundle(
    *,
    dataset: str,
    graph: str,
    variant: str,
    artifact: GraphArtifact,
    gold_standard_asset: Asset,
    split_asset: Asset,
    split_seed: int,
    feature_dim: int = 384,
    random_seed: int = 42,
    strict_coverage: bool = True,
) -> DataBundle:
    """Build a deterministic PyG graph with labels and predefined split masks."""
    if feature_dim < 1:
        raise ValueError("feature_dim must be >= 1")
    gold = _load_gold_standard(gold_standard_asset.path)
    nodes, relations, edge_count = _scan_graph(artifact.path)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    relation_to_index = {relation: index for index, relation in enumerate(relations)}
    gold_to_node = tuple(node_to_index.get(term) for term in gold.terms)
    missing_terms = tuple(
        term for term, node_index in zip(gold.terms, gold_to_node) if node_index is None
    )
    if strict_coverage and missing_terms:
        sample = ", ".join(repr(term) for term in missing_terms[:5])
        raise DataValidationError(
            f"Graph {dataset}/{graph} misses {len(missing_terms)} of {len(gold.terms)} "
            f"Gold Standard terms. Examples: {sample}"
        )

    edge_index, edge_type = _build_edges(
        artifact.path, node_to_index, relation_to_index, edge_count
    )
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    x = torch.rand((len(nodes), feature_dim), generator=generator, dtype=torch.float32)
    relation_x = torch.rand(
        (len(relations), feature_dim), generator=generator, dtype=torch.float32
    )

    label_names = tuple(sorted(set(gold.labels)))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    y = torch.full((len(nodes),), -1, dtype=torch.long)
    for term, label, node_index in zip(gold.terms, gold.labels, gold_to_node):
        if node_index is None:
            continue
        encoded = label_to_index[label]
        current = int(y[node_index])
        if current not in (-1, encoded):
            raise DataValidationError(
                f"Graph node {term!r} has conflicting Gold Standard labels"
            )
        y[node_index] = encoded

    masks = _build_split_masks(
        split_asset=split_asset,
        gold=gold,
        gold_to_node=gold_to_node,
        num_nodes=len(nodes),
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        train_mask=masks["train_idx"],
        val_mask=masks["val_idx"],
        test_mask=masks["test_idx"],
    )
    data.relation_x = relation_x
    data.num_classes = len(label_names)
    data.num_relations = len(relations)

    return DataBundle(
        dataset=dataset,
        graph=graph,
        variant=variant,
        split_seed=split_seed,
        artifact=artifact,
        data=data,
        node_to_index=node_to_index,
        index_to_node=nodes,
        relation_to_index=relation_to_index,
        index_to_relation=relations,
        label_names=label_names,
        gold_standard=gold,
        gold_to_node=gold_to_node,
        missing_gold_terms=missing_terms,
    )


def replace_bundle_split(
    bundle: DataBundle, *, split_asset: Asset, split_seed: int
) -> DataBundle:
    """Reuse a prepared graph while replacing only its predefined split masks."""
    masks = _build_split_masks(
        split_asset=split_asset,
        gold=bundle.gold_standard,
        gold_to_node=bundle.gold_to_node,
        num_nodes=bundle.data.num_nodes,
    )
    data = copy.copy(bundle.data)
    data.train_mask = masks["train_idx"]
    data.val_mask = masks["val_idx"]
    data.test_mask = masks["test_idx"]
    return replace(bundle, split_seed=split_seed, data=data)


def replace_bundle_random_features(
    bundle: DataBundle, *, random_seed: int
) -> DataBundle:
    """Reuse graph topology and deterministically replace random input features."""
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    data = copy.copy(bundle.data)
    data.x = torch.rand(
        (bundle.data.num_nodes, bundle.data.x.size(1)),
        generator=generator,
        dtype=torch.float32,
    )
    data.relation_x = torch.rand(
        (bundle.data.num_relations, bundle.data.relation_x.size(1)),
        generator=generator,
        dtype=torch.float32,
    )
    return replace(bundle, data=data)

