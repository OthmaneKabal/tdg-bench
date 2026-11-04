"""Lazy, checksummed Hugging Face backend for TDGBench datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from tdg_bench.datasets.local import LocalDatasetStore
from tdg_bench.datasets.manifest import Asset, DatasetManifest, GraphArtifact
from tdg_bench.exceptions import (
    DatasetDownloadError,
    DatasetNotFoundError,
    IntegrityError,
    InvalidManifestError,
    OptionalDependencyError,
)


DEFAULT_REPO_ID = "othmanekabal/TDGBench"
DEFAULT_REVISION = "v2.0.0"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str) -> str:
    return _SAFE_COMPONENT.sub("_", value).strip("_") or "default"


class HubDatasetStore:
    """Materialize only requested Hub assets into a stable local data root."""

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = DEFAULT_REVISION,
        cache_dir: str | Path | None = None,
        offline: bool = False,
    ) -> None:
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise ValueError("repo_id must be a non-empty string")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")
        if cache_dir is None:
            try:
                from platformdirs import user_cache_path
            except ImportError as exc:  # pragma: no cover
                raise OptionalDependencyError(
                    "Hugging Face loading requires platformdirs"
                ) from exc

            cache_base = user_cache_path("tdg-bench")
        else:
            cache_base = Path(cache_dir).expanduser().resolve()

        self.repo_id = repo_id.strip()
        self.revision = revision.strip()
        self.offline = bool(offline)
        self.cache_base = Path(cache_base)
        self.root = (
            self.cache_base
            / "datasets"
            / _safe_component(self.repo_id)
            / _safe_component(self.revision)
        ).resolve()
        self._hf_cache = (self.cache_base / "huggingface").resolve()
        self._local = LocalDatasetStore(self.root)
        self._root_assets: dict[str, Asset] | None = None
        self._verified: set[Path] = set()

    def _download(self, filename: str, *, force: bool = False) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError(
                "Automatic data download requires huggingface-hub"
            ) from exc

        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=str(self._hf_cache),
                local_dir=str(self.root),
                local_files_only=self.offline,
                force_download=force and not self.offline,
            )
        except Exception as exc:
            mode = "offline cache" if self.offline else "Hugging Face Hub"
            raise DatasetDownloadError(
                f"Could not retrieve {filename!r} from {mode} for "
                f"{self.repo_id}@{self.revision}: {exc}"
            ) from exc
        return Path(downloaded).resolve()

    def _root_manifest(self) -> Mapping[str, Any]:
        path = self.root / "manifest.json"
        if not path.is_file():
            self._download("manifest.json")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if self.offline:
                raise InvalidManifestError(
                    f"Could not read cached Hub manifest {path}: {exc}"
                ) from exc
            self._download("manifest.json", force=True)
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as retry_exc:
                raise InvalidManifestError(
                    f"Could not read Hub manifest {path}: {retry_exc}"
                ) from retry_exc
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise InvalidManifestError(f"Invalid root Hub manifest: {path}")
        datasets = raw.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            raise InvalidManifestError(
                f"Root Hub manifest has no dataset catalogue: {path}"
            )
        return datasets

    def _catalogue(self) -> dict[str, Asset]:
        if self._root_assets is None:
            self._root_assets = {
                dataset: Asset.from_dict(
                    self.root, value, f"datasets.{dataset}"
                )
                for dataset, value in self._root_manifest().items()
            }
        return self._root_assets

    def ensure_asset(self, asset: Asset) -> Asset:
        path = asset.path.resolve()
        try:
            hub_filename = path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise IntegrityError(
                f"Hub asset escapes the materialized data root: {path}"
            ) from exc
        if path in self._verified:
            return asset
        if not path.is_file():
            self._download(hub_filename)
        try:
            asset.verify()
        except IntegrityError:
            if self.offline:
                raise
            self._download(hub_filename, force=True)
            asset.verify()
        self._verified.add(path)
        return asset

    def available_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self._catalogue()))

    def manifest(self, dataset: str) -> DatasetManifest:
        catalogue = self._catalogue()
        if dataset not in catalogue:
            raise DatasetNotFoundError(
                f"Unknown dataset {dataset!r}. Available datasets: "
                f"{', '.join(sorted(catalogue))}."
            )
        self.ensure_asset(catalogue[dataset])
        return self._local.manifest(dataset)

    def available_graphs(self, dataset: str) -> tuple[str, ...]:
        self.manifest(dataset)
        return self._local.available_graphs(dataset)

    def available_variants(self, dataset: str, graph: str) -> tuple[str, ...]:
        self.manifest(dataset)
        return self._local.available_variants(dataset, graph)

    def graph(
        self,
        dataset: str,
        graph: str,
        *,
        variant: str = "default",
        verify: bool = True,
    ) -> GraphArtifact:
        self.manifest(dataset)
        artifact = self._local.graph(
            dataset, graph, variant=variant, verify=False
        )
        self.ensure_asset(artifact.asset)
        return artifact

    def available_corpora(self, dataset: str) -> tuple[str, ...]:
        self.manifest(dataset)
        return self._local.available_corpora(dataset)

    def corpus(self, dataset: str, corpus: str, *, verify: bool = True) -> Asset:
        self.manifest(dataset)
        asset = self._local.corpus(dataset, corpus, verify=False)
        self.ensure_asset(asset)
        return asset
