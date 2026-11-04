"""Lazy builders for the built-in TDGBench baseline architectures."""

from __future__ import annotations

from typing import Any

from tdg_bench.models.registry import register_model


@register_model("MLP")
def build_mlp(**kwargs: Any):
    from tdg_bench.models.encoders import NoGraphMLP

    return NoGraphMLP(**kwargs)


@register_model("GCN")
def build_gcn(**kwargs: Any):
    from tdg_bench.models.encoders import GCNEncoder

    return GCNEncoder(**kwargs)


@register_model("GAT")
def build_gat(**kwargs: Any):
    from tdg_bench.models.encoders import GATEncoder

    return GATEncoder(**kwargs)


@register_model("RGCN")
def build_rgcn(**kwargs: Any):
    from tdg_bench.models.encoders import RGCNEncoder

    return RGCNEncoder(**kwargs)


def _build_trans(*, scoring: str, variant: str, **kwargs: Any):
    from tdg_bench.models.encoders import TransGCNEncoder

    return TransGCNEncoder(scoring=scoring, variant=variant, **kwargs)


@register_model("TransEGCN_conv")
def build_transe_conv(**kwargs: Any):
    return _build_trans(scoring="TransE", variant="conv", **kwargs)


@register_model("TransEGCN_attn")
def build_transe_attn(**kwargs: Any):
    return _build_trans(scoring="TransE", variant="attn", **kwargs)


@register_model("RotatEGCN_conv")
def build_rotate_conv(**kwargs: Any):
    return _build_trans(scoring="RotatE", variant="conv", **kwargs)


@register_model("RotatEGCN_attn")
def build_rotate_attn(**kwargs: Any):
    return _build_trans(scoring="RotatE", variant="attn", **kwargs)

