from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from tdg_bench.evaluation.unified import (
    _PreparedUnifiedData,
    _aggregate_observations,
    run_unified_evaluation,
)


def test_unified_variance_separates_splits_and_training_seeds() -> None:
    summary = _aggregate_observations(
        [
            (42, 1, 1.0),
            (42, 2, 3.0),
            (123, 1, 5.0),
            (123, 2, 7.0),
        ]
    )
    assert summary.mean == pytest.approx(4.0)
    assert summary.std_split == pytest.approx(math.sqrt(8.0))
    assert summary.std_seed == pytest.approx(math.sqrt(2.0))
    assert summary.std_total == pytest.approx(math.sqrt(20.0 / 3.0))
    assert summary.n_splits == 2
    assert summary.n_seeds_per_split == 2
    assert summary.n_runs == 4
    assert summary.by_split[42].mean == pytest.approx(2.0)
    assert summary.by_split[123].mean == pytest.approx(6.0)


def _tiny_prepared_data() -> _PreparedUnifiedData:
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
    bundle = SimpleNamespace(
        dataset="example",
        graph="tdg",
        variant="default",
        split_seed=42,
        initial_embedding_model=None,
        label_names=("class-0", "class-1"),
        artifact=SimpleNamespace(asset=SimpleNamespace(sha256="a" * 64)),
        data=data,
    )
    return _PreparedUnifiedData(bundle, {42: bundle}, (42,))


@pytest.mark.gnn
def test_unified_evaluation_resumes_completed_runs(tmp_path) -> None:
    benchmark = SimpleNamespace(data_root=tmp_path / "benchmark_data")
    prepared = _tiny_prepared_data()
    arguments = {
        "benchmark": benchmark,
        "dataset": "example",
        "graph": "tdg",
        "model": "GCN",
        "hidden_dims": (8,),
        "training_seeds": (1,),
        "max_epochs": 1,
        "patience": 1,
        "device": "cpu",
        "metrics": ("accuracy", "f1_macro"),
        "class_wise": False,
        "track_training": True,
        "verbose": False,
        "results_dir": tmp_path / "results",
        "_prepared_data": prepared,
    }

    first = run_unified_evaluation(**arguments)
    second = run_unified_evaluation(**arguments)

    assert first.completed_runs == 1
    assert first.resumed_runs == 0
    assert second.completed_runs == 0
    assert second.resumed_runs == 1
    assert second.runs == first.runs
    assert list((tmp_path / "results").rglob("training_seed_*.json"))
    assert first.export_dir is not None
    assert (first.export_dir / "protocol.json").is_file()
    assert (first.export_dir / "runs.csv").is_file()
    assert (first.export_dir / "summary.csv").is_file()
    assert (first.export_dir / "class_metrics.csv").is_file()
    assert (first.export_dir / "confusion_matrices.json").is_file()
    history_path = first.export_dir / "training_history.csv"
    assert history_path.is_file()
    assert "train_loss" in history_path.read_text(encoding="utf-8")
