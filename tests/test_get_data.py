from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("torch_geometric")

from tdg_bench import TDGBench, check_new_graph
from tdg_bench.exceptions import DataValidationError


def metadata(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def make_dataset(tmp_path: Path, *, leaking_duplicate: bool = False) -> Path:
    root = tmp_path / "example"
    graph = root / "graphs" / "tdg" / "default.json"
    gold = root / "gold_standard" / "gold_standard.xlsx"
    split = root / "splits" / "split_42.json"
    graph.parent.mkdir(parents=True)
    gold.parent.mkdir(parents=True)
    split.parent.mkdir(parents=True)
    graph.write_text(
        json.dumps(
            [
                {"subject": "alpha", "predicate": "related", "object": "beta"},
                {"subject": "beta", "predicate": "is_a", "object": "gamma"},
                {"subject": "alpha", "predicate": "related", "object": "gamma"},
            ]
        ),
        encoding="utf-8",
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["term", "label"])
    sheet.append(["alpha", "A"])
    sheet.append(["beta", "B"])
    sheet.append(["alpha" if leaking_duplicate else "gamma", "A"])
    workbook.save(gold)
    split.write_text(
        json.dumps({"train_idx": [0], "val_idx": [1], "test_idx": [2]}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset": {"id": "example", "title": "Example", "version": "test"},
        "gold_standard": metadata(gold, root),
        "splits": {"42": metadata(split, root)},
        "graphs": {"tdg": {"default": metadata(graph, root)}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.mark.gnn
def test_get_data_builds_deterministic_pyg_bundle(tmp_path: Path) -> None:
    bench = TDGBench(make_dataset(tmp_path))
    first = bench.get_data("example", "tdg", feature_dim=8, random_seed=7)
    second = bench.get_data("example", "tdg", feature_dim=8, random_seed=7)
    assert first.data.num_nodes == 3
    assert first.data.num_edges == 3
    assert first.data.num_relations == 2
    assert first.label_names == ("A", "B")
    assert first.split_sizes == {"train": 1, "validation": 1, "test": 1}
    assert first.coverage == 1.0
    assert torch.equal(first.data.x, second.data.x)
    assert torch.equal(first.data.edge_index, second.data.edge_index)


@pytest.mark.gnn
def test_get_no_graph_data_reads_only_gold_standard_and_split(tmp_path: Path) -> None:
    data_root = make_dataset(tmp_path)
    graph_path = data_root / "example" / "graphs" / "tdg" / "default.json"
    graph_path.unlink()
    bench = TDGBench(data_root)

    bundle = bench.get_no_graph_data("example", feature_dim=8, feature_seed=7)

    assert bundle.graph == "no_graph"
    assert bundle.variant == "none"
    assert bundle.artifact is None
    assert bundle.index_to_node == ("alpha", "beta", "gamma")
    assert bundle.data.num_nodes == 3
    assert bundle.data.num_edges == 0
    assert bundle.data.num_relations == 0
    assert bundle.split_sizes == {"train": 1, "validation": 1, "test": 1}


@pytest.mark.gnn
def test_evaluate_mlp_requires_no_graph(tmp_path: Path) -> None:
    bench = TDGBench(make_dataset(tmp_path))
    result = bench.evaluate(
        "example",
        model="MLP",
        feature_dim=8,
        hidden_dim=8,
        max_epochs=2,
        patience=1,
        device="cpu",
    )
    assert result.model == "MLP"
    assert result.graph == "no_graph"
    assert result.training_mode == "full_batch"

    with pytest.raises(ValueError, match="does not accept a graph"):
        bench.evaluate("example", "tdg", model="MLP")


@pytest.mark.gnn
def test_unified_mlp_uses_common_nodes_without_graph(tmp_path: Path) -> None:
    data_root = make_dataset(tmp_path)
    (data_root / "example" / "graphs" / "tdg" / "default.json").unlink()
    bench = TDGBench(data_root)

    result = bench.unified_evaluation(
        dataset="example",
        model="MLP",
        hidden_dims=[8],
        split_seeds=[42],
        training_seeds=[1],
        feature_dim=8,
        max_epochs=1,
        patience=1,
        device="cpu",
        persist=False,
        verbose=False,
    )

    assert result.graph == "no_graph"
    assert result.model == "MLP"
    assert result.runs[0].training_mode == "full_batch"
    assert result.runs[0].num_neighbors == ()


@pytest.mark.gnn
def test_unified_suite_combines_graph_model_and_no_graph_mlp(tmp_path: Path) -> None:
    bench = TDGBench(make_dataset(tmp_path))
    suite = bench.unified_evaluation(
        dataset="example",
        graph="tdg",
        models=["GCN", "MLP"],
        hidden_dims=[8],
        split_seeds=[42],
        training_seeds=[1],
        feature_dim=8,
        num_neighbors=[-1, -1],
        max_epochs=1,
        patience=1,
        device="cpu",
        persist=False,
        verbose=False,
    )

    assert suite.graph == "tdg"
    assert suite.results["GCN"].runs[0].training_mode == "neighbor_sampling"
    assert suite.results["MLP"].graph == "no_graph"
    assert suite.results["MLP"].runs[0].training_mode == "full_batch"


@pytest.mark.gnn
def test_get_data_rejects_same_node_in_multiple_partitions(tmp_path: Path) -> None:
    bench = TDGBench(make_dataset(tmp_path, leaking_duplicate=True))
    with pytest.raises(DataValidationError, match="same graph node"):
        bench.get_data("example", "tdg", feature_dim=4)


@pytest.mark.gnn
def test_get_data_accepts_user_graph_without_manifest_change(tmp_path: Path) -> None:
    data_root = make_dataset(tmp_path / "data")
    custom_graph = tmp_path / "my_graph.json"
    custom_graph.write_text(
        json.dumps(
            [
                {"subject": "alpha", "predicate": "custom", "object": "beta"},
                {"subject": "beta", "predicate": "custom", "object": "gamma"},
            ]
        ),
        encoding="utf-8",
    )
    bench = TDGBench(data_root)
    bundle = bench.get_data(
        "example",
        graph_path=custom_graph,
        graph_name="user_tdg",
        feature_dim=4,
    )
    assert bundle.graph == "user_tdg"
    assert bundle.artifact.path == custom_graph.resolve()
    assert bundle.coverage == 1.0


@pytest.mark.gnn
def test_get_data_accepts_user_csv_graph(tmp_path: Path) -> None:
    data_root = make_dataset(tmp_path / "data")
    custom_graph = tmp_path / "my_graph.csv"
    custom_graph.write_text(
        "subject,predicate,object,score\n"
        "alpha,custom,beta,0.9\n"
        "beta,custom,gamma,0.8\n",
        encoding="utf-8",
    )
    bench = TDGBench(data_root)
    bundle = bench.get_data(
        "example",
        graph_path=custom_graph,
        graph_name="user_csv",
        feature_dim=4,
    )
    assert bundle.graph == "user_csv"
    assert bundle.data.num_nodes == 3
    assert bundle.data.num_edges == 2
    assert bundle.coverage == 1.0


def test_check_new_graph_reports_common_node_coverage(tmp_path: Path) -> None:
    data_root = make_dataset(tmp_path / "data")
    custom_graph = tmp_path / "partial_graph.csv"
    custom_graph.write_text(
        "subject,predicate,object\nalpha,related,beta\n",
        encoding="utf-8",
    )
    bench = TDGBench(data_root)

    report = bench.check_new_graph(
        dataset="example",
        graph_path=custom_graph,
        graph_name="partial_method",
    )

    assert report.graph_name == "partial_method"
    assert report.graph_nodes == 2
    assert report.graph_edges == 1
    assert report.common_nodes_total == 3
    assert report.common_nodes_covered == 2
    assert report.common_nodes_missing == 1
    assert report.missing_common_nodes == ("gamma",)
    assert report.coverage == pytest.approx(2 / 3)
    assert report.is_complete is False
    assert report.to_dict()["coverage"] == pytest.approx(2 / 3)

    standalone = check_new_graph(
        custom_graph,
        data_root / "example" / "gold_standard" / "gold_standard.xlsx",
        dataset="example",
        graph_name="standalone_method",
    )
    assert standalone.common_nodes_covered == 2
    assert standalone.missing_common_nodes == ("gamma",)
    assert standalone.graph_name == "standalone_method"


@pytest.mark.gnn
def test_evaluate_many_reuses_prepared_graph(tmp_path: Path) -> None:
    bench = TDGBench(make_dataset(tmp_path))
    summary = bench.evaluate_many(
        "example",
        "tdg",
        model="GCN",
        split_seeds=[42],
        random_seeds=[1, 2],
        feature_dim=4,
        hidden_dim=4,
        max_epochs=1,
        patience=1,
        device="cpu",
    )
    assert len(summary.runs) == 2
    assert {run.random_seed for run in summary.runs} == {1, 2}
