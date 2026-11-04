"""Model registry and built-in TDGBench architectures."""

from tdg_bench.models.registry import (
    MODEL_REGISTRY,
    ModelRegistry,
    available_model_names,
    build_model,
    register_model,
)

# Encoder modules remain lazily imported by each registered builder.
from tdg_bench.models import builtins as _builtins  # noqa: E402,F401

__all__ = [
    "MODEL_REGISTRY",
    "ModelRegistry",
    "available_model_names",
    "build_model",
    "register_model",
]
