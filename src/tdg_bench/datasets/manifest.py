"""Typed representation of a local TDGBench dataset manifest."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tdg_bench.exceptions import IntegrityError, InvalidManifestError


SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InvalidManifestError(f"{label} must be a JSON object")
    return value


def _resolve_asset_path(dataset_root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise InvalidManifestError("Asset path must be a non-empty string")
    candidate = (dataset_root / relative_path).resolve()
    root = dataset_root.resolve()
    if not candidate.is_relative_to(root):
        raise InvalidManifestError(f"Asset path escapes the dataset directory: {relative_path}")
    return candidate


@dataclass(frozen=True)
class Asset:
    """One immutable file declared by a dataset manifest."""

    path: Path
    relative_path: str
    size: int
    sha256: str | None

    @classmethod
    def from_local_path(cls, path: str | Path) -> "Asset":
        """Create an unchecksummed asset for a user-provided local file."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise IntegrityError(f"Local asset is missing or is not a file: {resolved}")
        return cls(
            path=resolved,
            relative_path=resolved.name,
            size=resolved.stat().st_size,
            sha256=None,
        )

    @classmethod
    def from_dict(cls, dataset_root: Path, value: object, label: str) -> "Asset":
        data = _require_mapping(value, label)
        relative_path = data.get("path")
        size = data.get("size")
        sha256 = data.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise InvalidManifestError(f"{label}.size must be an integer >= 0")
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256.lower()) is None:
            raise InvalidManifestError(f"{label}.sha256 must be a 64-character SHA-256")
        return cls(
            path=_resolve_asset_path(dataset_root, relative_path),
            relative_path=relative_path,
            size=size,
            sha256=sha256.lower(),
        )

    def verify(self) -> None:
        """Verify existence, byte size and SHA-256."""
        if not self.path.is_file():
            raise IntegrityError(f"Dataset asset is missing: {self.path}")
        actual_size = self.path.stat().st_size
        if actual_size != self.size:
            raise IntegrityError(
                f"Size mismatch for {self.path}: expected {self.size}, got {actual_size}"
            )
        if self.sha256 is not None:
            digest = hashlib.sha256()
            with self.path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != self.sha256:
                raise IntegrityError(
                    f"SHA-256 mismatch for {self.path}: expected {self.sha256}, "
                    f"got {actual_sha256}"
                )


@dataclass(frozen=True)
class GraphArtifact:
    """A resolved graph file and its benchmark identity."""

    dataset: str
    graph: str
    variant: str
    asset: Asset

    @classmethod
    def from_local_path(
        cls,
        *,
        dataset: str,
        graph: str,
        path: str | Path,
        variant: str = "default",
    ) -> "GraphArtifact":
        """Resolve a user graph without adding it to the official manifest."""
        return cls(dataset, graph, variant, Asset.from_local_path(path))

    @property
    def path(self) -> Path:
        return self.asset.path

    def verify(self) -> "GraphArtifact":
        self.asset.verify()
        return self


@dataclass(frozen=True)
class DatasetManifest:
    """Files and graph variants belonging to one benchmark dataset."""

    dataset_id: str
    title: str
    version: str
    root: Path
    gold_standard: Asset
    splits: Mapping[int, Asset]
    graphs: Mapping[str, Mapping[str, Asset]]
    corpus: Mapping[str, Asset]

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        manifest_path = Path(path)
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidManifestError(f"Could not read manifest {manifest_path}: {exc}") from exc
        data = _require_mapping(raw, "manifest")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise InvalidManifestError(
                f"Unsupported manifest schema_version={data.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        dataset = _require_mapping(data.get("dataset"), "dataset")
        dataset_id = dataset.get("id")
        title = dataset.get("title")
        version = dataset.get("version")
        if not all(isinstance(value, str) and value.strip() for value in (dataset_id, title, version)):
            raise InvalidManifestError("dataset.id, dataset.title and dataset.version are required")

        root = manifest_path.parent.resolve()
        gold_standard = Asset.from_dict(root, data.get("gold_standard"), "gold_standard")

        split_data = _require_mapping(data.get("splits"), "splits")
        splits: dict[int, Asset] = {}
        for seed_text, asset_data in split_data.items():
            try:
                seed = int(seed_text)
            except (TypeError, ValueError) as exc:
                raise InvalidManifestError(f"Invalid split seed: {seed_text!r}") from exc
            if seed in splits:
                raise InvalidManifestError(f"Duplicate split seed: {seed}")
            splits[seed] = Asset.from_dict(root, asset_data, f"splits.{seed}")
        if not splits:
            raise InvalidManifestError("At least one split is required")

        graph_data = _require_mapping(data.get("graphs"), "graphs")
        graphs: dict[str, dict[str, Asset]] = {}
        for graph_name, variants_data in graph_data.items():
            if not isinstance(graph_name, str) or not graph_name.strip():
                raise InvalidManifestError("Graph names must be non-empty strings")
            variants_mapping = _require_mapping(variants_data, f"graphs.{graph_name}")
            variants: dict[str, Asset] = {}
            for variant_name, asset_data in variants_mapping.items():
                if not isinstance(variant_name, str) or not variant_name.strip():
                    raise InvalidManifestError("Variant names must be non-empty strings")
                variants[variant_name] = Asset.from_dict(
                    root, asset_data, f"graphs.{graph_name}.{variant_name}"
                )
            if "default" not in variants:
                raise InvalidManifestError(f"Graph {graph_name!r} has no default variant")
            graphs[graph_name] = variants
        if not graphs:
            raise InvalidManifestError("At least one graph is required")

        corpus_data = data.get("corpus", {})
        corpus_mapping = _require_mapping(corpus_data, "corpus")
        corpus: dict[str, Asset] = {}
        for corpus_name, asset_data in corpus_mapping.items():
            if not isinstance(corpus_name, str) or not corpus_name.strip():
                raise InvalidManifestError("Corpus names must be non-empty strings")
            corpus[corpus_name] = Asset.from_dict(
                root, asset_data, f"corpus.{corpus_name}"
            )

        return cls(
            dataset_id=dataset_id,
            title=title,
            version=version,
            root=root,
            gold_standard=gold_standard,
            splits=splits,
            graphs=graphs,
            corpus=corpus,
        )

