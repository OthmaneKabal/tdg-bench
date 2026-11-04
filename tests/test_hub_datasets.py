from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tdg_bench import TDGBench


def _metadata(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fake_hub_release(root: Path) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    dataset = root / "example"
    graph = dataset / "graphs" / "tdg" / "default.json"
    gold = dataset / "gold_standard" / "gold_standard.xlsx"
    split = dataset / "splits" / "split_42.json"
    corpus = dataset / "corpus" / "documents.json"
    graph.parent.mkdir(parents=True)
    gold.parent.mkdir(parents=True)
    split.parent.mkdir(parents=True)
    corpus.parent.mkdir(parents=True)
    graph.write_text(
        json.dumps(
            [
                {"subject": "alpha", "predicate": "r", "object": "beta"},
                {"subject": "beta", "predicate": "r", "object": "gamma"},
            ]
        ),
        encoding="utf-8",
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["term", "label"])
    sheet.append(["alpha", "A"])
    sheet.append(["beta", "B"])
    sheet.append(["gamma", "A"])
    workbook.save(gold)
    workbook.close()
    split.write_text(
        json.dumps({"train_idx": [0], "val_idx": [1], "test_idx": [2]}),
        encoding="utf-8",
    )
    corpus.write_text('{"1":{"text":"example"}}', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": {"id": "example", "title": "Example", "version": "1"},
        "gold_standard": _metadata(gold, dataset),
        "splits": {"42": _metadata(split, dataset)},
        "graphs": {"tdg": {"default": _metadata(graph, dataset)}},
        "corpus": {"documents": _metadata(corpus, dataset)},
    }
    dataset_manifest = dataset / "manifest.json"
    dataset_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    root_manifest = {
        "schema_version": 1,
        "release": "1",
        "datasets": {"example": _metadata(dataset_manifest, root)},
    }
    (root / "manifest.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    return root


@pytest.mark.gnn
def test_hub_backend_downloads_requested_assets_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    remote = _fake_hub_release(tmp_path / "remote")
    calls: list[str] = []

    def fake_download(*, filename: str, local_dir: str, **_kwargs) -> str:
        calls.append(filename)
        source = remote / filename
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    bench = TDGBench(
        repo_id="example/TDGBench",
        revision="v1",
        cache_dir=tmp_path / "cache",
    )

    assert bench.source == "hub"
    assert bench.available_datasets() == ("example",)
    assert calls == ["manifest.json"]
    assert bench.available_graphs("example") == ("tdg",)
    assert calls[-1] == "example/manifest.json"

    artifact = bench.get_graph("example", "tdg")
    assert artifact.path.is_file()
    assert calls[-1] == "example/graphs/tdg/default.json"

    corpus = bench.get_corpus("example", "documents")
    assert corpus.path.is_file()
    assert calls[-1] == "example/corpus/documents.json"

    bundle = bench.get_data(
        "example",
        "tdg",
        split_seed=42,
        feature_dim=4,
    )
    assert bundle.coverage == 1.0
    assert bundle.data.num_edges == 2
    assert "example/gold_standard/gold_standard.xlsx" in calls
    assert "example/splits/split_42.json" in calls

    calls_before_offline = tuple(calls)
    offline_bench = TDGBench(
        repo_id="example/TDGBench",
        revision="v1",
        cache_dir=tmp_path / "cache",
        offline=True,
    )
    offline_bundle = offline_bench.get_data(
        "example",
        "tdg",
        split_seed=42,
        feature_dim=4,
    )
    assert offline_bundle.coverage == 1.0
    assert tuple(calls) == calls_before_offline
