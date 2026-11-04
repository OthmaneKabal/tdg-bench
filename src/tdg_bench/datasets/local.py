"""Local filesystem backend for TDGBench datasets."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path

from tdg_bench.datasets.manifest import DatasetManifest, GraphArtifact
from tdg_bench.exceptions import (
    CorpusNotFoundError,
    DatasetNotFoundError,
    GraphNotFoundError,
)


class LocalDatasetStore:
    """Discover benchmark datasets stored below one local root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._manifests: dict[str, DatasetManifest] = {}

    def ensure_asset(self, asset: Asset) -> Asset:
        """Consume local assets in place without forcing checksum verification."""
        return asset

    def available_datasets(self) -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(
            sorted(
                directory.name
                for directory in self.root.iterdir()
                if directory.is_dir() and (directory / "manifest.json").is_file()
            )
        )

    def manifest(self, dataset: str) -> DatasetManifest:
        if dataset in self._manifests:
            return self._manifests[dataset]
        available = self.available_datasets()
        if dataset not in available:
            suggestion = get_close_matches(dataset, available, n=1, cutoff=0.55)
            message = f"Unknown dataset {dataset!r}. Available datasets: {', '.join(available) or 'none'}."
            if suggestion:
                message += f" Did you mean {suggestion[0]!r}?"
            raise DatasetNotFoundError(message)
        manifest = DatasetManifest.load(self.root / dataset / "manifest.json")
        if manifest.dataset_id != dataset:
            raise DatasetNotFoundError(
                f"Dataset directory {dataset!r} contains manifest id {manifest.dataset_id!r}"
            )
        self._manifests[dataset] = manifest
        return manifest

    def available_graphs(self, dataset: str) -> tuple[str, ...]:
        return tuple(sorted(self.manifest(dataset).graphs))

    def available_variants(self, dataset: str, graph: str) -> tuple[str, ...]:
        manifest = self.manifest(dataset)
        if graph not in manifest.graphs:
            available = tuple(sorted(manifest.graphs))
            raise GraphNotFoundError(
                f"Unknown graph {graph!r} for dataset {dataset!r}. "
                f"Available graphs: {', '.join(available)}."
            )
        return tuple(sorted(manifest.graphs[graph]))

    def available_corpora(self, dataset: str) -> tuple[str, ...]:
        return tuple(sorted(self.manifest(dataset).corpus))

    def corpus(self, dataset: str, corpus: str, *, verify: bool = True):
        manifest = self.manifest(dataset)
        if corpus not in manifest.corpus:
            available = tuple(sorted(manifest.corpus))
            raise CorpusNotFoundError(
                f"Unknown corpus {corpus!r} for dataset {dataset!r}. "
                f"Available corpora: {', '.join(available) or 'none'}."
            )
        asset = manifest.corpus[corpus]
        if verify:
            asset.verify()
        return asset

    def graph(
        self,
        dataset: str,
        graph: str,
        *,
        variant: str = "default",
        verify: bool = True,
    ) -> GraphArtifact:
        manifest = self.manifest(dataset)
        if graph not in manifest.graphs:
            available = tuple(sorted(manifest.graphs))
            suggestion = get_close_matches(graph, available, n=1, cutoff=0.55)
            message = (
                f"Unknown graph {graph!r} for dataset {dataset!r}. "
                f"Available graphs: {', '.join(available)}."
            )
            if suggestion:
                message += f" Did you mean {suggestion[0]!r}?"
            raise GraphNotFoundError(message)
        variants = manifest.graphs[graph]
        if variant not in variants:
            raise GraphNotFoundError(
                f"Unknown variant {variant!r} for {dataset}/{graph}. "
                f"Available variants: {', '.join(sorted(variants))}."
            )
        artifact = GraphArtifact(dataset, graph, variant, variants[variant])
        return artifact.verify() if verify else artifact

