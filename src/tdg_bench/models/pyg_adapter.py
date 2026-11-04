"""Automatic node-classification wrapper for PyG convolution classes."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing


_ACTIVATIONS = {
    "relu": F.relu,
    "elu": F.elu,
    "gelu": F.gelu,
}


class PyGConvolutionClassifier(nn.Module):
    """Stack one PyG ``MessagePassing`` type and attach a classifier head."""

    def __init__(
        self,
        convolution: type[MessagePassing],
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_relations: int,
        num_layers: int,
        dropout: float,
        conv_kwargs: Mapping[str, Any] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if not inspect.isclass(convolution) or not issubclass(convolution, MessagePassing):
            raise TypeError("convolution must be a PyG MessagePassing class")
        if input_dim < 1 or hidden_dim < 1 or num_classes < 2:
            raise ValueError("input_dim, hidden_dim and num_classes must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}. Available: {sorted(_ACTIVATIONS)}"
            )

        supplied = dict(conv_kwargs or {})
        reserved = {"in_channels", "out_channels", "num_relations"} & set(supplied)
        if reserved:
            raise ValueError(
                "TDGBench controls convolution dimensions; remove these keys from "
                f"conv_kwargs: {sorted(reserved)}"
            )
        signature = inspect.signature(convolution)
        parameters = signature.parameters
        if "in_channels" not in parameters or "out_channels" not in parameters:
            raise TypeError(
                f"{convolution.__name__} cannot be constructed automatically: its "
                "constructor must declare in_channels and out_channels. Provide a "
                "complete TDGBench-compatible model class for this convolution."
            )
        if "num_relations" in parameters and num_relations < 1:
            raise ValueError(f"{convolution.__name__} requires graph relation types")
        if "concat" in parameters and "concat" not in supplied:
            supplied["concat"] = False

        self.convs = nn.ModuleList()
        current_dim = input_dim
        for _ in range(num_layers):
            arguments = {
                "in_channels": current_dim,
                "out_channels": hidden_dim,
                **supplied,
            }
            if "num_relations" in parameters:
                arguments["num_relations"] = num_relations
            conv = convolution(**arguments)
            self.convs.append(conv)
            current_dim = hidden_dim
            if bool(arguments.get("concat", False)):
                current_dim *= int(arguments.get("heads", 1))

        forward_parameters = inspect.signature(self.convs[0].forward).parameters
        self.uses_edge_type = "edge_type" in forward_parameters
        self.uses_edge_attr = (
            "edge_attr" in forward_parameters and "edge_dim" in supplied
        )
        self.activation = _ACTIVATIONS[activation]
        self.dropout = dropout
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x, edge_index, edge_type=None, relation_x=None):
        for index, conv in enumerate(self.convs):
            if self.uses_edge_type:
                if edge_type is None:
                    raise ValueError("This PyG convolution requires edge_type")
                x = conv(x, edge_index, edge_type)
            elif self.uses_edge_attr:
                if edge_type is None or relation_x is None:
                    raise ValueError(
                        "This PyG convolution requires per-edge relation features"
                    )
                x = conv(x, edge_index, relation_x[edge_type])
            else:
                x = conv(x, edge_index)
            if index + 1 < len(self.convs):
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def is_pyg_convolution_type(value: object) -> bool:
    return inspect.isclass(value) and issubclass(value, MessagePassing)


__all__ = ["PyGConvolutionClassifier", "is_pyg_convolution_type"]
