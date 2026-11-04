"""Deterministic Sentence-Transformer feature initialisation with local caching."""

from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path

from tdg_bench.exceptions import DataValidationError, OptionalDependencyError
from tdg_bench.data_preparation.builder import DataBundle

try:
    import torch
except ImportError as exc:  # pragma: no cover - broken installation guard
    raise OptionalDependencyError(
        "Text embeddings require PyTorch. Reinstall with: pip install --upgrade tdg-bench"
    ) from exc


DEFAULT_SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"


def _load_encoder(initial_embedding_model: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - broken installation guard
        raise OptionalDependencyError(
            "Sentence-BERT features require sentence-transformers. "
            "Reinstall with: pip install --upgrade tdg-bench"
        ) from exc
    try:
        return SentenceTransformer(
            initial_embedding_model,
            device=device,
            local_files_only=True,
        )
    except OSError:
        # First use may legitimately require a download. Once cached, the
        # offline branch above avoids network checks in later experiments.
        return SentenceTransformer(initial_embedding_model, device=device)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for embeddings but is not available")
    return device


def _digest_strings(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cache_path(
    bundle: DataBundle, initial_embedding_model: str, cache_dir: Path
) -> tuple[Path, str, str]:
    node_digest = _digest_strings(bundle.index_to_node)
    relation_digest = _digest_strings(bundle.index_to_relation)
    identity = hashlib.sha256(
        f"{initial_embedding_model}\0{node_digest}\0{relation_digest}".encode("utf-8")
    ).hexdigest()
    safe_model_name = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", initial_embedding_model
    ).strip("_")
    directory = cache_dir / bundle.dataset / bundle.graph / bundle.variant
    return directory / f"{safe_model_name}-{identity[:16]}.pt", node_digest, relation_digest


def _valid_cache(
    payload: object,
    *,
    initial_embedding_model: str,
    node_digest: str,
    relation_digest: str,
    num_nodes: int,
    num_relations: int,
) -> bool:
    if not isinstance(payload, dict):
        return False
    nodes = payload.get("node_features")
    relations = payload.get("relation_features")
    return (
        payload.get("schema_version") == 1
        and payload.get(
            "initial_embedding_model", payload.get("model_name")
        ) == initial_embedding_model
        and payload.get("node_digest") == node_digest
        and payload.get("relation_digest") == relation_digest
        and isinstance(nodes, torch.Tensor)
        and isinstance(relations, torch.Tensor)
        and nodes.ndim == 2
        and relations.ndim == 2
        and nodes.size(0) == num_nodes
        and relations.size(0) == num_relations
        and nodes.size(1) == relations.size(1)
    )


def with_sentence_transformer_features(
    bundle: DataBundle,
    *,
    initial_embedding_model: str = DEFAULT_SENTENCE_BERT,
    cache_dir: str | Path,
    batch_size: int = 256,
    device: str = "auto",
    use_cache: bool = True,
    show_progress: bool = True,
) -> DataBundle:
    """Replace random graph features with deterministic text embeddings."""
    if (
        not isinstance(initial_embedding_model, str)
        or not initial_embedding_model.strip()
    ):
        raise ValueError("initial_embedding_model must be a non-empty string")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    cache_path, node_digest, relation_digest = _cache_path(
        bundle, initial_embedding_model, resolved_cache_dir
    )
    payload = None
    if use_cache and cache_path.is_file():
        try:
            candidate = torch.load(cache_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError):
            candidate = None
        if _valid_cache(
            candidate,
            initial_embedding_model=initial_embedding_model,
            node_digest=node_digest,
            relation_digest=relation_digest,
            num_nodes=bundle.data.num_nodes,
            num_relations=bundle.data.num_relations,
        ):
            payload = candidate
            if show_progress:
                print(f"Sentence-BERT cache loaded: {cache_path}", flush=True)

    if payload is None:
        resolved_device = _resolve_device(device)
        if show_progress:
            print(
                f"Encoding {bundle.data.num_nodes:,} nodes and "
                f"{bundle.data.num_relations:,} relations with "
                f"{initial_embedding_model} "
                f"on {resolved_device}...",
                flush=True,
            )
        encoder = _load_encoder(initial_embedding_model, resolved_device)
        encoder.eval()
        node_features = encoder.encode(
            list(bundle.index_to_node),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            normalize_embeddings=False,
        ).detach().cpu().to(torch.float32).contiguous()
        relation_features = (
            encoder.encode(
                list(bundle.index_to_relation),
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                normalize_embeddings=False,
            ).detach().cpu().to(torch.float32).contiguous()
            if bundle.index_to_relation
            else torch.empty((0, node_features.size(1)), dtype=torch.float32)
        )
        if node_features.size(1) != relation_features.size(1):
            raise DataValidationError(
                "Node and relation embeddings have different dimensions"
            )
        payload = {
            "schema_version": 1,
            "initial_embedding_model": initial_embedding_model,
            "node_digest": node_digest,
            "relation_digest": relation_digest,
            "node_features": node_features,
            "relation_features": relation_features,
        }
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(f".tmp-{os.getpid()}.pt")
            torch.save(payload, temporary)
            temporary.replace(cache_path)
            if show_progress:
                print(f"Sentence-BERT cache written: {cache_path}", flush=True)

    data = copy.copy(bundle.data)
    data.x = payload["node_features"]
    data.relation_x = payload["relation_features"]
    return replace(
        bundle,
        data=data,
        initial_embedding_model=initial_embedding_model,
        embedding_cache_path=cache_path if use_cache else None,
    )
