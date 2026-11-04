"""High-level TDGBench facade."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tdg_bench.datasets.hub import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    HubDatasetStore,
)
from tdg_bench.datasets.local import LocalDatasetStore
from tdg_bench.datasets.manifest import GraphArtifact
from tdg_bench.datasets.variants import (
    BIOMED_PREPROCESSING_OPTIONS,
    CURATED_UMLS_PREPROCESSING_OPTIONS,
    resolve_preprocessing_variant,
)


class TDGBench:
    """Access TDGBench from the versioned Hub release or a local folder."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = DEFAULT_REVISION,
        cache_dir: str | Path | None = None,
        offline: bool = False,
    ) -> None:
        if data_root is None:
            store = HubDatasetStore(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                offline=offline,
            )
            self.data_root = store.root
            self._store = store
            self.source = "hub"
        else:
            self.data_root = Path(data_root).expanduser().resolve()
            self._store = LocalDatasetStore(self.data_root)
            self.source = "local"
        self.repo_id = repo_id
        self.revision = revision
        self.offline = bool(offline)
        self._curated_umls_dir = self.data_root / "biomed" / "graphs" / "cg"

    def available_datasets(self) -> tuple[str, ...]:
        """Return benchmark dataset identifiers from the selected backend."""
        return self._store.available_datasets()

    def available_graphs(self, dataset: str) -> tuple[str, ...]:
        """Return graph identifiers available for one dataset."""
        available = set(self._store.available_graphs(dataset))
        if dataset == "biomed" and (self._curated_umls_dir / "default.json").is_file():
            available.add("cg")
        return tuple(sorted(available))

    def available_variants(self, dataset: str, graph: str) -> tuple[str, ...]:
        """Return the immutable variants registered for one graph."""
        if dataset == "biomed" and graph == "cg":
            if not (self._curated_umls_dir / "default.json").is_file():
                from tdg_bench.exceptions import GraphNotFoundError

                raise GraphNotFoundError(
                    "Biomed curated graph 'cg' has not been built. "
                    "Call bench.build_curated_umls(...) first."
                )
            reserved = {
                "build_metadata",
                "inverse_relations",
                "semantic_type_integrations",
            }
            return tuple(
                sorted(
                    path.stem
                    for path in self._curated_umls_dir.glob("*.json")
                    if path.stem not in reserved
                    and not path.stem.endswith(".summary")
                )
            )
        return self._store.available_variants(dataset, graph)

    def available_preprocessing_options(
        self, dataset: str, graph: str
    ) -> tuple[str, ...]:
        """Return public preprocessing options, or an empty tuple for DBpedia."""
        if dataset == "biomed" and graph == "cg":
            return CURATED_UMLS_PREPROCESSING_OPTIONS
        variants = set(self.available_variants(dataset, graph))
        if dataset != "biomed":
            return ()
        return tuple(
            option
            for option in BIOMED_PREPROCESSING_OPTIONS
            if resolve_preprocessing_variant(dataset, options=(option,)) in variants
        )

    def available_corpora(self, dataset: str) -> tuple[str, ...]:
        """Return corpus artifacts registered for one benchmark domain."""
        return self._store.available_corpora(dataset)

    def get_corpus(self, dataset: str, corpus: str, *, verify: bool = True):
        """Resolve one corpus asset and optionally verify its checksum."""
        return self._store.corpus(dataset, corpus, verify=verify)

    def available_models(self) -> tuple[str, ...]:
        """Return the stable names of built-in graph and no-graph models."""
        from tdg_bench.models import available_model_names

        return available_model_names()

    def available_metrics(self) -> tuple[str, ...]:
        """Return classification metric names accepted by ``evaluate``."""
        from tdg_bench.evaluation.metrics import available_metric_names

        return available_metric_names()

    def check_new_graph(
        self,
        dataset: str,
        graph_path: str | Path,
        *,
        graph_name: str | None = None,
    ):
        """Report common-node coverage before evaluating a user graph."""
        manifest = self._store.manifest(dataset)
        self._store.ensure_asset(manifest.gold_standard)
        from tdg_bench.graphs import check_new_graph

        return check_new_graph(
            graph_path,
            manifest.gold_standard.path,
            dataset=dataset,
            graph_name=graph_name,
        )

    def get_graph(
        self,
        dataset: str,
        graph: str,
        *,
        variant: str = "default",
        options: Iterable[str] | None = None,
        verify: bool = True,
    ) -> GraphArtifact:
        """Resolve one graph artifact and optionally verify its integrity."""
        resolved_options = None if options is None else tuple(options)
        resolved_variant = resolve_preprocessing_variant(
            dataset,
            graph=graph,
            variant=variant,
            options=resolved_options,
        )
        if dataset == "biomed" and graph == "cg":
            path = self._curated_umls_dir / f"{resolved_variant}.json"
            if not path.is_file() and resolved_options:
                from tdg_bench.data_preparation.biomed_curated import (
                    materialize_biomed_curated_variant,
                )

                path = materialize_biomed_curated_variant(
                    self._curated_umls_dir, resolved_options
                )
            artifact = GraphArtifact.from_local_path(
                dataset=dataset,
                graph=graph,
                path=path,
                variant=resolved_variant,
            )
            return artifact.verify() if verify else artifact
        return self._store.graph(
            dataset, graph, variant=resolved_variant, verify=verify
        )

    def build_curated_umls(
        self,
        *,
        db_host: str,
        db_user: str,
        db_password: str,
        db_name: str,
        db_port: int = 3306,
        umls_release: str = "unknown",
        path: str | Path | None = None,
        source: str = "NCI",
        min_relation_frequency: int = 50,
        semantic_type_fraction: float = 0.01,
        random_seed: int = 42,
        fetch_size: int = 10_000,
        metadata_chunk_size: int = 1_000,
        verbose: bool = True,
        progress_every: int = 100_000,
    ):
        """Build local Biomed ``cg`` raw/default graphs from UMLS MySQL.

        ``path`` is an optional TDGBench data root. The generated files are
        always placed below ``biomed/graphs/cg`` and become immediately
        available through :meth:`get_graph` and :meth:`get_data`.
        """
        from tdg_bench.data_preparation.biomed_curated import (
            BiomedCuratedBuildConfig,
            UMLSDatabaseConfig,
            build_biomed_curated,
        )

        destination_root = (
            self.data_root
            if path is None
            else Path(path).expanduser().resolve()
        )
        graph_dir = destination_root / "biomed" / "graphs" / "cg"
        result = build_biomed_curated(
            database_config=UMLSDatabaseConfig(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
            ),
            output_dir=graph_dir,
            build_config=BiomedCuratedBuildConfig(
                source=source,
                umls_release=umls_release,
                min_relation_frequency=min_relation_frequency,
                semantic_type_fraction=semantic_type_fraction,
                random_seed=random_seed,
                fetch_size=fetch_size,
                metadata_chunk_size=metadata_chunk_size,
            ),
            verbose=verbose,
            progress_every=progress_every,
        )
        self._curated_umls_dir = graph_dir
        return result

    def get_data(
        self,
        dataset: str,
        graph: str | None = None,
        *,
        graph_path: str | Path | None = None,
        graph_name: str | None = None,
        split_seed: int = 42,
        variant: str = "default",
        options: Iterable[str] | None = None,
        feature_dim: int = 384,
        initial_embedding_model: str | None = None,
        embedding_batch_size: int = 256,
        embedding_device: str = "auto",
        embedding_cache: bool = True,
        embedding_progress: bool = True,
        embedding_cache_dir: str | Path | None = None,
        random_seed: int = 42,
        feature_seed: int | None = None,
        strict_coverage: bool = True,
        verify_graph: bool = False,
    ):
        """Prepare one graph and predefined split as a PyTorch Geometric bundle.

        Features are deterministic random tensors by default. Set
        ``initial_embedding_model`` to initialise nodes and relations with a cached
        Sentence-Transformer model instead.
        """
        manifest = self._store.manifest(dataset)
        if split_seed not in manifest.splits:
            available = ", ".join(str(seed) for seed in sorted(manifest.splits))
            raise ValueError(
                f"Unknown split_seed={split_seed} for dataset {dataset!r}. "
                f"Available seeds: {available}."
            )
        self._store.ensure_asset(manifest.gold_standard)
        self._store.ensure_asset(manifest.splits[split_seed])
        if graph_path is None:
            if graph is None:
                raise ValueError("Provide either graph or graph_path")
            if graph_name is not None:
                raise ValueError("graph_name is only valid with graph_path")
            resolved_graph_name = graph
            artifact = self.get_graph(
                dataset,
                graph,
                variant=variant,
                options=options,
                verify=verify_graph,
            )
        else:
            if options is not None and tuple(options):
                raise ValueError(
                    "Preprocessing options select published graph variants and "
                    "cannot be used with graph_path"
                )
            if graph is not None:
                raise ValueError("graph and graph_path are mutually exclusive")
            path = Path(graph_path).expanduser().resolve()
            resolved_graph_name = graph_name or path.stem
            if not resolved_graph_name.strip():
                raise ValueError("graph_name must be a non-empty string")
            artifact = GraphArtifact.from_local_path(
                dataset=dataset,
                graph=resolved_graph_name,
                path=path,
                variant=variant,
            )
            if verify_graph:
                artifact.verify()
        from tdg_bench.data_preparation.builder import build_data_bundle

        resolved_feature_seed = random_seed if feature_seed is None else feature_seed
        bundle = build_data_bundle(
            dataset=dataset,
            graph=resolved_graph_name,
            variant=artifact.variant,
            artifact=artifact,
            gold_standard_asset=manifest.gold_standard,
            split_asset=manifest.splits[split_seed],
            split_seed=split_seed,
            feature_dim=feature_dim,
            random_seed=resolved_feature_seed,
            strict_coverage=strict_coverage,
        )
        if initial_embedding_model is not None:
            from tdg_bench.data_preparation.embeddings import (
                with_sentence_transformer_features,
            )

            cache_dir = (
                Path(embedding_cache_dir)
                if embedding_cache_dir is not None
                else self.data_root / ".cache" / "embeddings"
            )
            bundle = with_sentence_transformer_features(
                bundle,
                initial_embedding_model=initial_embedding_model,
                cache_dir=cache_dir,
                batch_size=embedding_batch_size,
                device=embedding_device,
                use_cache=embedding_cache,
                show_progress=embedding_progress,
            )
        return bundle

    def get_no_graph_data(
        self,
        dataset: str,
        *,
        split_seed: int = 42,
        feature_dim: int = 384,
        initial_embedding_model: str | None = None,
        embedding_batch_size: int = 256,
        embedding_device: str = "auto",
        embedding_cache: bool = True,
        embedding_progress: bool = True,
        embedding_cache_dir: str | Path | None = None,
        random_seed: int = 42,
        feature_seed: int | None = None,
    ):
        """Prepare common nodes, labels and a split without reading a graph."""
        manifest = self._store.manifest(dataset)
        if split_seed not in manifest.splits:
            available = ", ".join(str(seed) for seed in sorted(manifest.splits))
            raise ValueError(
                f"Unknown split_seed={split_seed} for dataset {dataset!r}. "
                f"Available seeds: {available}."
            )
        self._store.ensure_asset(manifest.gold_standard)
        self._store.ensure_asset(manifest.splits[split_seed])
        from tdg_bench.data_preparation.builder import build_no_graph_bundle

        resolved_feature_seed = random_seed if feature_seed is None else feature_seed
        bundle = build_no_graph_bundle(
            dataset=dataset,
            gold_standard_asset=manifest.gold_standard,
            split_asset=manifest.splits[split_seed],
            split_seed=split_seed,
            feature_dim=feature_dim,
            random_seed=resolved_feature_seed,
        )
        if initial_embedding_model is not None:
            from tdg_bench.data_preparation.embeddings import (
                with_sentence_transformer_features,
            )

            cache_dir = (
                Path(embedding_cache_dir)
                if embedding_cache_dir is not None
                else self.data_root / ".cache" / "embeddings"
            )
            bundle = with_sentence_transformer_features(
                bundle,
                initial_embedding_model=initial_embedding_model,
                cache_dir=cache_dir,
                batch_size=embedding_batch_size,
                device=embedding_device,
                use_cache=embedding_cache,
                show_progress=embedding_progress,
            )
        return bundle

    def evaluate(
        self,
        dataset: str,
        graph: str | None = None,
        model: Any = "GCN",
        *,
        graph_path: str | Path | None = None,
        graph_name: str | None = None,
        split_seed: int = 42,
        variant: str = "default",
        options: Iterable[str] | None = None,
        feature_dim: int = 384,
        initial_embedding_model: str | None = None,
        embedding_batch_size: int = 256,
        embedding_device: str = "auto",
        embedding_cache: bool = True,
        embedding_progress: bool = True,
        embedding_cache_dir: str | Path | None = None,
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
        feature_seed: int = 42,
        device: str = "auto",
        strict_coverage: bool = True,
        verify_graph: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        metrics: str | tuple[str, ...] | list[str] | None = None,
        selection_metric: str = "f1_macro",
        class_wise: bool = False,
        track_training: bool = False,
        verbose: bool = False,
        log_every: int = 10,
    ):
        """Train and evaluate one built-in or custom model on one fixed split."""
        no_graph = isinstance(model, str) and model == "MLP"
        if no_graph:
            if graph is not None or graph_path is not None or graph_name is not None:
                raise ValueError("model='MLP' does not accept a graph")
            if options is not None and tuple(options):
                raise ValueError("model='MLP' does not accept graph preprocessing options")
            bundle = self.get_no_graph_data(
                dataset,
                split_seed=split_seed,
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
            bundle = self.get_data(
                dataset,
                graph,
                graph_path=graph_path,
                graph_name=graph_name,
                split_seed=split_seed,
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
        from tdg_bench.evaluation import evaluate_bundle

        return evaluate_bundle(
            bundle,
            model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            shuffle=shuffle,
            num_workers=num_workers,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            random_seed=random_seed,
            device=device,
            model_kwargs=model_kwargs,
            metrics=metrics,
            selection_metric=selection_metric,
            class_wise=class_wise,
            track_training=track_training,
            verbose=verbose,
            log_every=log_every,
        )

    def evaluate_many(
        self,
        dataset: str,
        graph: str | None = None,
        model: Any = "GCN",
        *,
        split_seeds: tuple[int, ...] | list[int] | None = None,
        random_seeds: tuple[int, ...] | list[int] = (42,),
        **kwargs: Any,
    ):
        """Evaluate several predefined splits/initialisations and aggregate them."""
        manifest = self._store.manifest(dataset)
        selected_splits = (
            tuple(sorted(manifest.splits)) if split_seeds is None else tuple(split_seeds)
        )
        selected_random_seeds = tuple(random_seeds)
        if not selected_splits:
            raise ValueError("split_seeds must not be empty")
        if not selected_random_seeds:
            raise ValueError("random_seeds must not be empty")

        unknown_splits = set(selected_splits) - set(manifest.splits)
        if unknown_splits:
            available = ", ".join(str(seed) for seed in sorted(manifest.splits))
            raise ValueError(
                f"Unknown split seeds {sorted(unknown_splits)} for {dataset!r}. "
                f"Available seeds: {available}."
            )
        for split_seed in selected_splits:
            self._store.ensure_asset(manifest.splits[split_seed])

        no_graph = isinstance(model, str) and model == "MLP"
        if no_graph and graph is not None:
            raise ValueError("model='MLP' does not accept a graph")
        data_loader = self.get_no_graph_data if no_graph else self.get_data
        data_parameter_names = set(inspect.signature(data_loader).parameters)
        data_parameter_names -= {"self", "dataset", "graph", "split_seed", "random_seed"}
        data_kwargs = {
            name: value for name, value in kwargs.items() if name in data_parameter_names
        }
        evaluation_kwargs = {
            name: value for name, value in kwargs.items() if name not in data_parameter_names
        }
        first_split = selected_splits[0]
        base_bundle = (
            self.get_no_graph_data(
                dataset,
                split_seed=first_split,
                **data_kwargs,
            )
            if no_graph
            else self.get_data(
                dataset,
                graph,
                split_seed=first_split,
                **data_kwargs,
            )
        )

        from tdg_bench.evaluation import aggregate_results, evaluate_bundle
        from tdg_bench.data_preparation import replace_bundle_split

        results = []
        for random_seed in selected_random_seeds:
            for split_seed in selected_splits:
                run_bundle = (
                    base_bundle
                    if split_seed == first_split
                    else replace_bundle_split(
                        base_bundle,
                        split_asset=manifest.splits[split_seed],
                        split_seed=split_seed,
                    )
                )
                results.append(
                    evaluate_bundle(
                        run_bundle,
                        copy.deepcopy(model),
                        random_seed=random_seed,
                        **evaluation_kwargs,
                    )
                )

        return aggregate_results(results)

    def unified_evaluation(
        self,
        dataset: str,
        graph: str | None = None,
        model: Any | None = None,
        *,
        models: tuple[Any, ...] | list[Any] | None = None,
        **kwargs: Any,
    ):
        """Evaluate one, several, or every registered model.

        The protocol separates within-split training-seed variability from
        between-split variability and selects hidden size using validation
        metrics only.
        """
        from tdg_bench.evaluation.unified import (
            run_unified_evaluation,
            run_unified_suite,
        )

        if model is not None and models is not None:
            raise ValueError("Use either model or models, not both")
        if model is not None:
            return run_unified_evaluation(
                self,
                dataset=dataset,
                graph=graph,
                model=model,
                **kwargs,
            )
        if models is None:
            from tdg_bench.models import available_model_names

            models = list(available_model_names())

        return run_unified_suite(
            self,
            dataset=dataset,
            graph=graph,
            models=models,
            **kwargs,
        )

