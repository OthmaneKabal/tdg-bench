from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tdg_bench import TDGBench
from tdg_bench.exceptions import (
    CorpusNotFoundError,
    DatasetNotFoundError,
    GraphNotFoundError,
    IntegrityError,
)


def metadata(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    dataset = tmp_path / "example"
    graph = dataset / "graphs" / "tdg" / "default.json"
    gold = dataset / "gold_standard" / "gold_standard.xlsx"
    split = dataset / "splits" / "split_42.json"
    graph.parent.mkdir(parents=True)
    gold.parent.mkdir(parents=True)
    split.parent.mkdir(parents=True)
    graph.write_text('[{"subject":"a","predicate":"r","object":"b"}]', encoding="utf-8")
    gold.write_bytes(b"test-gold-standard")
    split.write_text('{"train_idx":[0],"val_idx":[],"test_idx":[]}', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": {"id": "example", "title": "Example", "version": "test"},
        "gold_standard": metadata(gold, dataset),
        "splits": {"42": metadata(split, dataset)},
        "graphs": {"tdg": {"default": metadata(graph, dataset)}},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_discovers_and_resolves_local_graph(data_root: Path) -> None:
    bench = TDGBench(data_root)
    assert bench.available_datasets() == ("example",)
    assert bench.available_graphs("example") == ("tdg",)
    graph = bench.get_graph("example", "tdg")
    assert graph.path.name == "default.json"
    assert graph.dataset == "example"


def test_unknown_names_are_informative(data_root: Path) -> None:
    bench = TDGBench(data_root)
    with pytest.raises(DatasetNotFoundError, match="Available datasets: example"):
        bench.available_graphs("missing")
    with pytest.raises(GraphNotFoundError, match="Available graphs: tdg"):
        bench.get_graph("example", "missing")


def test_integrity_mismatch_is_detected(data_root: Path) -> None:
    bench = TDGBench(data_root)
    graph = bench.get_graph("example", "tdg", verify=False)
    graph.path.write_text("corrupted", encoding="utf-8")
    with pytest.raises(IntegrityError, match="Size mismatch|SHA-256 mismatch"):
        graph.verify()


def test_dataset_without_corpus_reports_none(data_root: Path) -> None:
    bench = TDGBench(data_root)
    assert bench.available_corpora("example") == ()
    with pytest.raises(CorpusNotFoundError, match="Available corpora: none"):
        bench.get_corpus("example", "missing")
