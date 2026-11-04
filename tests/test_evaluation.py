from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, RGCNConv, SAGEConv

from tdg_bench.evaluation import aggregate_results, evaluate_bundle
from tdg_bench.models import available_model_names


def tiny_bundle():
    data = Data(
        x=torch.rand(6, 8),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 0, 3], [1, 2, 0, 4, 5, 3, 3, 0]]
        ),
        edge_type=torch.tensor([0, 1, 0, 1, 0, 1, 1, 0]),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
    )
    data.relation_x = torch.rand(2, 8)
    data.num_classes = 2
    data.num_relations = 2
    return SimpleNamespace(
        dataset="example",
        graph="tdg",
        variant="default",
        split_seed=42,
        data=data,
    )


class CustomGNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x, edge_index, edge_type, relation_x):
        del edge_index, edge_type, relation_x
        return self.linear(x)


@pytest.mark.gnn
def test_evaluate_builtin_model_and_aggregate(capsys) -> None:
    first = evaluate_bundle(
        tiny_bundle(),
        "GCN",
        hidden_dim=8,
        max_epochs=3,
        patience=2,
        random_seed=7,
        device="cpu",
        verbose=True,
        log_every=1,
        metrics="all",
        class_wise=True,
        track_training=True,
    )
    second = evaluate_bundle(
        tiny_bundle(),
        "GCN",
        hidden_dim=8,
        max_epochs=2,
        patience=1,
        random_seed=8,
        device="cpu",
    )
    summary = aggregate_results((first, second))
    assert first.model == "GCN"
    assert first.test.support == 2
    assert 0.0 <= first.test.macro_f1 <= 1.0
    assert "balanced_accuracy" in first.test.values
    assert "matthews_corrcoef" in first.test.values
    assert first.test.per_class is not None
    assert len(first.test.per_class) == 2
    assert first.test.confusion_matrix is not None
    assert len(summary.runs) == 2
    assert "macro_f1_mean" in summary.aggregate["test"]
    assert len(first.history) == first.epochs_ran
    assert first.history[0].train_accuracy is not None
    assert first.history[0].train_values is not None
    assert second.history == ()
    assert "val_macro_f1=" in capsys.readouterr().out


@pytest.mark.gnn
def test_evaluate_custom_model_factory() -> None:
    result = evaluate_bundle(
        tiny_bundle(),
        CustomGNN,
        max_epochs=2,
        patience=1,
        random_seed=11,
        device="cpu",
    )
    assert result.model == "CustomGNN"
    assert result.validation.support == 2


@pytest.mark.gnn
@pytest.mark.parametrize("model_name", available_model_names())
def test_every_builtin_model_supports_neighbor_sampled_batches(model_name) -> None:
    result = evaluate_bundle(
        tiny_bundle(),
        model_name,
        hidden_dim=8,
        batch_size=2,
        num_neighbors=(-1, -1),
        max_epochs=1,
        patience=1,
        random_seed=13,
        device="cpu",
    )
    assert result.model == model_name
    assert result.training_mode == "neighbor_sampling"
    assert result.batch_size == 2
    assert result.num_neighbors == (-1, -1)


@pytest.mark.gnn
@pytest.mark.parametrize("convolution", (GCNConv, SAGEConv, GATConv, RGCNConv))
def test_pyg_convolution_class_is_wrapped_automatically(convolution) -> None:
    result = evaluate_bundle(
        tiny_bundle(),
        convolution,
        hidden_dim=8,
        num_layers=2,
        batch_size=2,
        num_neighbors=(-1, -1),
        max_epochs=1,
        patience=1,
        random_seed=17,
        device="cpu",
    )
    assert result.model == convolution.__name__
    assert result.training_mode == "neighbor_sampling"


@pytest.mark.gnn
def test_pyg_convolution_accepts_nested_convolution_kwargs() -> None:
    result = evaluate_bundle(
        tiny_bundle(),
        GATConv,
        hidden_dim=8,
        num_layers=2,
        model_kwargs={"conv_kwargs": {"heads": 2, "concat": False}},
        batch_size=2,
        num_neighbors=(-1, -1),
        max_epochs=1,
        patience=1,
        device="cpu",
    )
    assert result.model == "GATConv"

