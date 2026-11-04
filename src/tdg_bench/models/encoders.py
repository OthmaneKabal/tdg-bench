"""Built-in graph encoders used by the TDGBench evaluation runner."""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import GATConv, GCNConv, RGCNConv
    from torch_geometric.utils import scatter, softmax
except ImportError as exc:  # pragma: no cover - optional dependency guard
    from tdg_bench.exceptions import OptionalDependencyError

    raise OptionalDependencyError(
        "Built-in models require the GNN dependencies. Install them with: "
        "pip install 'tdg-bench[gnn]'"
    ) from exc


def _validate_dimensions(
    input_dim: int, hidden_dim: int, num_layers: int, dropout: float
) -> None:
    if input_dim < 1 or hidden_dim < 1:
        raise ValueError("input_dim and hidden_dim must be >= 1")
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


class GCNEncoder(nn.Module):
    """Multi-layer GCN encoder with a stable TDGBench forward contract."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        *,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _validate_dimensions(input_dim, hidden_dim, num_layers, dropout)
        dimensions = [input_dim, *([hidden_dim] * num_layers)]
        self.convs = nn.ModuleList(
            GCNConv(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )
        self.dropout = dropout
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, edge_type=None, relation_x=None):
        del edge_type, relation_x
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index + 1 < len(self.convs):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """Multi-head GAT encoder whose heads are averaged at every layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        *,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _validate_dimensions(input_dim, hidden_dim, num_layers, dropout)
        if heads < 1:
            raise ValueError("heads must be >= 1")
        dimensions = [input_dim, *([hidden_dim] * num_layers)]
        self.convs = nn.ModuleList(
            GATConv(
                dimensions[index],
                dimensions[index + 1],
                heads=heads,
                concat=False,
                dropout=dropout,
            )
            for index in range(num_layers)
        )
        self.dropout = dropout
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, edge_type=None, relation_x=None):
        del edge_type, relation_x
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index + 1 < len(self.convs):
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RGCNEncoder(nn.Module):
    """Relational GCN encoder using the graph's predicate identifiers."""

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        hidden_dim: int = 64,
        *,
        num_layers: int = 2,
        num_bases: int | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _validate_dimensions(input_dim, hidden_dim, num_layers, dropout)
        if num_relations < 1:
            raise ValueError("num_relations must be >= 1")
        dimensions = [input_dim, *([hidden_dim] * num_layers)]
        if num_bases is None:
            num_bases = min(30, num_relations)
        self.convs = nn.ModuleList(
            RGCNConv(
                dimensions[index],
                dimensions[index + 1],
                num_relations=num_relations,
                num_bases=num_bases,
            )
            for index in range(num_layers)
        )
        self.dropout = dropout
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, edge_type=None, relation_x=None):
        del relation_x
        if edge_type is None:
            raise ValueError("RGCN requires edge_type")
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if index + 1 < len(self.convs):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class _TransLayer(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        relation_input_dim: int,
        output_dim: int,
        *,
        scoring: str,
        variant: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Linear(node_input_dim, output_dim, bias=False)
        self.relation_projection = nn.Linear(relation_input_dim, output_dim, bias=False)
        self.self_projection = nn.Linear(node_input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.scoring = scoring
        self.variant = variant
        self.dropout = dropout
        self.output_dim = output_dim

    def _rotate(self, nodes, relations, *, inverse: bool = False):
        node_real, node_imag = nodes.chunk(2, dim=-1)
        relation_real, relation_imag = relations.chunk(2, dim=-1)
        relation_norm = torch.sqrt(
            relation_real.square() + relation_imag.square()
        ).clamp_min(1e-9)
        relation_real = relation_real / relation_norm
        relation_imag = relation_imag / relation_norm
        if inverse:
            relation_imag = -relation_imag
        return torch.cat(
            (
                node_real * relation_real - node_imag * relation_imag,
                node_real * relation_imag + node_imag * relation_real,
            ),
            dim=-1,
        )

    def forward(self, x, edge_index, edge_type, relation_x):
        source, target = edge_index
        nodes = self.node_projection(x)
        relations = self.relation_projection(relation_x)
        edge_relations = relations[edge_type]

        if self.scoring == "TransE":
            forward_messages = nodes[source] + edge_relations
            reverse_messages = nodes[target] - edge_relations
        else:
            forward_messages = self._rotate(nodes[source], edge_relations)
            reverse_messages = self._rotate(nodes[target], edge_relations, inverse=True)

        destinations = torch.cat((target, source))
        messages = torch.cat((forward_messages, reverse_messages))
        if self.variant == "attn":
            queries = nodes[destinations]
            scores = (messages * queries).sum(dim=-1) / math.sqrt(self.output_dim)
            messages = messages * softmax(
                scores, destinations, num_nodes=nodes.size(0)
            ).unsqueeze(-1)
            aggregated = scatter(
                messages,
                destinations,
                dim=0,
                dim_size=nodes.size(0),
                reduce="sum",
            )
        else:
            aggregated = scatter(
                messages,
                destinations,
                dim=0,
                dim_size=nodes.size(0),
                reduce="mean",
            )

        output = self.normalization(self.self_projection(x) + aggregated)
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output, relations


class TransGCNEncoder(nn.Module):
    """Relation-aware encoder with TransE or RotatE message composition.

    ``variant='conv'`` mean-aggregates bidirectional messages. ``'attn'``
    learns a local attention distribution from message/destination similarity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        *,
        num_layers: int = 2,
        scoring: str = "TransE",
        variant: str = "conv",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _validate_dimensions(input_dim, hidden_dim, num_layers, dropout)
        if scoring not in {"TransE", "RotatE"}:
            raise ValueError("scoring must be 'TransE' or 'RotatE'")
        if variant not in {"conv", "attn"}:
            raise ValueError("variant must be 'conv' or 'attn'")
        if scoring == "RotatE" and hidden_dim % 2:
            raise ValueError("RotatE requires an even hidden_dim")

        node_dimensions = [input_dim, *([hidden_dim] * num_layers)]
        relation_dimensions = [input_dim, *([hidden_dim] * num_layers)]
        self.layers = nn.ModuleList(
            _TransLayer(
                node_dimensions[index],
                relation_dimensions[index],
                hidden_dim,
                scoring=scoring,
                variant=variant,
                dropout=dropout,
            )
            for index in range(num_layers)
        )
        self.output_dim = hidden_dim
        self.scoring = scoring
        self.variant = variant

    def forward(self, x, edge_index, edge_type=None, relation_x=None):
        if edge_type is None or relation_x is None:
            raise ValueError("TransE/RotatE encoders require edge_type and relation_x")
        for layer in self.layers:
            x, relation_x = layer(x, edge_index, edge_type, relation_x)
        return x


class NodeClassifier(nn.Module):
    """Attach a node-classification head to a TDGBench encoder."""

    def __init__(self, encoder: nn.Module, num_classes: int) -> None:
        super().__init__()
        output_dim = getattr(encoder, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim < 1:
            raise ValueError("An encoder must expose a positive integer output_dim")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.encoder = encoder
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x, edge_index, edge_type=None, relation_x=None):
        embeddings = self.encoder(x, edge_index, edge_type, relation_x)
        return self.classifier(embeddings)

