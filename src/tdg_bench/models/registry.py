"""Explicit registry for built-in and user-provided graph encoders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ModelBuilder = Callable[..., Any]


class ModelRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, ModelBuilder] = {}

    def register(self, name: str, builder: ModelBuilder) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must be a non-empty string")
        if name in self._builders:
            raise ValueError(f"A model is already registered as {name!r}")
        self._builders[name] = builder

    def names(self) -> tuple[str, ...]:
        return tuple(self._builders)

    def build(self, name: str, **kwargs: Any) -> Any:
        try:
            builder = self._builders[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model {name!r}. Available models: {', '.join(self.names())}."
            ) from exc
        return builder(**kwargs)


MODEL_REGISTRY = ModelRegistry()


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model builder under a stable public name."""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        MODEL_REGISTRY.register(name, builder)
        return builder

    return decorator


def available_model_names() -> tuple[str, ...]:
    return MODEL_REGISTRY.names()


def build_model(name: str, **kwargs: Any) -> Any:
    return MODEL_REGISTRY.build(name, **kwargs)

