from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from tdg_bench.data_preparation.builder import DataBundle, GoldStandard
from tdg_bench.data_preparation.embeddings import with_sentence_transformer_features


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def eval(self):
        return self

    def encode(self, texts, **kwargs):
        del kwargs
        self.calls += 1
        return torch.arange(len(texts) * 3, dtype=torch.float32).reshape(len(texts), 3)


def make_bundle() -> DataBundle:
    data = Data(
        x=torch.rand(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_type=torch.tensor([0, 0]),
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, False, False]),
        val_mask=torch.tensor([False, True, False]),
        test_mask=torch.tensor([False, False, True]),
    )
    data.relation_x = torch.rand(1, 2)
    data.num_classes = 2
    data.num_relations = 1
    return DataBundle(
        dataset="example",
        graph="tdg",
        variant="default",
        split_seed=42,
        artifact=None,
        data=data,
        node_to_index={"a": 0, "b": 1, "c": 2},
        index_to_node=("a", "b", "c"),
        relation_to_index={"r": 0},
        index_to_relation=("r",),
        label_names=("A", "B"),
        gold_standard=GoldStandard(("a", "b", "c"), ("A", "B", "A")),
        gold_to_node=(0, 1, 2),
        missing_gold_terms=(),
    )


@pytest.mark.gnn
def test_sentence_transformer_features_are_cached(tmp_path: Path, monkeypatch) -> None:
    encoder = FakeEncoder()
    monkeypatch.setattr(
        "tdg_bench.data_preparation.embeddings._load_encoder",
        lambda initial_embedding_model, device: encoder,
    )
    first = with_sentence_transformer_features(
        make_bundle(),
        initial_embedding_model="example/model",
        cache_dir=tmp_path,
        device="cpu",
        show_progress=False,
    )
    assert first.data.x.shape == (3, 3)
    assert first.data.relation_x.shape == (1, 3)
    assert first.initial_embedding_model == "example/model"
    assert first.embedding_cache_path is not None
    assert first.embedding_cache_path.is_file()
    assert encoder.calls == 2

    monkeypatch.setattr(
        "tdg_bench.data_preparation.embeddings._load_encoder",
        lambda initial_embedding_model, device: (
            _ for _ in ()
        ).throw(AssertionError("cache missed")),
    )
    second = with_sentence_transformer_features(
        make_bundle(),
        initial_embedding_model="example/model",
        cache_dir=tmp_path,
        device="cpu",
        show_progress=False,
    )
    assert torch.equal(first.data.x, second.data.x)
