from __future__ import annotations

import json
import gzip
from pathlib import Path

import pytest

from tdg_bench.exceptions import DataValidationError
from tdg_bench.graphs import iter_graph_records, iter_json_array


def test_iter_json_array_streams_across_small_chunks(tmp_path: Path) -> None:
    records = [
        {"subject": "node " + "x" * 2000, "predicate": "r", "object": str(index)}
        for index in range(4)
    ]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    assert list(iter_json_array(path, chunk_size=1024)) == records


def test_iter_json_array_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text('{"subject":"a"}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="top-level JSON array"):
        list(iter_json_array(path))


def test_iter_graph_records_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "graph.jsonl"
    path.write_text(
        '{"subject":"a","predicate":"r","object":"b"}\n'
        '{"subject":"b","predicate":"r","object":"c"}\n',
        encoding="utf-8",
    )
    assert list(iter_graph_records(path)) == [
        {"subject": "a", "predicate": "r", "object": "b"},
        {"subject": "b", "predicate": "r", "object": "c"},
    ]


@pytest.mark.parametrize(("extension", "delimiter"), [("csv", ","), ("tsv", "\t")])
def test_iter_graph_records_reads_delimited_tables(
    tmp_path: Path, extension: str, delimiter: str
) -> None:
    path = tmp_path / f"graph.{extension}"
    path.write_text(
        delimiter.join(("Subject", "Predicate", "Object", "confidence"))
        + "\n"
        + delimiter.join(("a", "r", "b", "0.9"))
        + "\n",
        encoding="utf-8",
    )
    records = list(iter_graph_records(path))
    assert records[0]["subject"] == "a"
    assert records[0]["predicate"] == "r"
    assert records[0]["object"] == "b"
    assert records[0]["confidence"] == "0.9"


def test_iter_graph_records_reads_gzip_csv(tmp_path: Path) -> None:
    path = tmp_path / "graph.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("subject,predicate,object\na,r,b\n")
    assert list(iter_graph_records(path))[0]["object"] == "b"


def test_iter_graph_records_reads_xlsx(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "graph.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["subject", "predicate", "object"])
    sheet.append(["a", "r", "b"])
    workbook.save(path)
    workbook.close()
    assert list(iter_graph_records(path)) == [
        {"subject": "a", "predicate": "r", "object": "b"}
    ]


def test_iter_graph_records_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "graph.csv"
    path.write_text("source,relation,target\na,r,b\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="missing required columns"):
        list(iter_graph_records(path))


def test_iter_graph_records_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "graph.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(DataValidationError, match="Unsupported graph format"):
        list(iter_graph_records(path))

