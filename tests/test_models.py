from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tdg_bench.models import available_model_names, build_model
from tdg_bench.models.encoders import NodeClassifier


GRAPH_MODELS = (
    "GCN",
    "GAT",
    "RGCN",
    "TransEGCN_conv",
    "TransEGCN_attn",
    "RotatEGCN_conv",
    "RotatEGCN_attn",
)
EXPECTED_MODELS = ("MLP", *GRAPH_MODELS)


@pytest.mark.gnn
def test_all_builtin_models_share_one_forward_contract() -> None:
    assert available_model_names() == EXPECTED_MODELS
    x = torch.rand(5, 8)
    relation_x = torch.rand(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_type = torch.tensor([0, 1, 0, 1])

    for name in GRAPH_MODELS:
        kwargs = {
            "input_dim": 8,
            "hidden_dim": 6,
            "num_layers": 2,
            "dropout": 0.0,
        }
        if name == "RGCN":
            kwargs["num_relations"] = 2
        encoder = build_model(name, **kwargs)
        classifier = NodeClassifier(encoder, num_classes=3)
        logits = classifier(x, edge_index, edge_type, relation_x)
        assert logits.shape == (5, 3), name

    mlp = build_model(
        "MLP", input_dim=8, hidden_dim=6, num_classes=3, dropout=0.0
    )
    assert mlp(x).shape == (5, 3)


@pytest.mark.gnn
def test_relation_aware_models_require_relation_inputs() -> None:
    encoder = build_model(
        "TransEGCN_conv", input_dim=8, hidden_dim=6, num_layers=1
    )
    with pytest.raises(ValueError, match="require edge_type and relation_x"):
        encoder(torch.rand(3, 8), torch.tensor([[0], [1]]))

