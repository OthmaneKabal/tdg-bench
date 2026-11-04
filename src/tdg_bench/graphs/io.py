"""Memory-bounded readers for graph files."""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tdg_bench.exceptions import DataValidationError


REQUIRED_GRAPH_COLUMNS = ("subject", "predicate", "object")
SUPPORTED_GRAPH_FORMATS = ("json", "jsonl", "ndjson", "csv", "tsv", "xlsx")


def _format_from_path(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    extension = suffixes[-1].lstrip(".") if suffixes else ""
    if extension not in SUPPORTED_GRAPH_FORMATS:
        supported = ", ".join(f".{name}" for name in SUPPORTED_GRAPH_FORMATS)
        raise DataValidationError(
            f"Unsupported graph format for {path}. Supported extensions: "
            f"{supported}; text formats may also use .gz."
        )
    if path.suffix.lower() == ".gz" and extension == "xlsx":
        raise DataValidationError("Compressed .xlsx.gz graph files are not supported")
    return extension


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8-sig", newline="")
    return path.open(encoding="utf-8-sig", newline="")


def _column_lookup(columns: list[object], path: Path) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for column in columns:
        if column is None:
            continue
        normalized = str(column).strip().casefold()
        if normalized in lookup:
            raise DataValidationError(
                f"Graph {path} contains duplicate column {normalized!r}"
            )
        lookup[normalized] = column
    missing = set(REQUIRED_GRAPH_COLUMNS) - set(lookup)
    if missing:
        raise DataValidationError(
            f"Graph {path} is missing required columns: {sorted(missing)}. "
            f"Required columns: {list(REQUIRED_GRAPH_COLUMNS)}."
        )
    return lookup


def iter_json_array(path: str | Path, *, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Yield values from a top-level JSON array without loading it all in RAM."""
    if chunk_size < 1024:
        raise ValueError("chunk_size must be >= 1024")
    source = Path(path)
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False

    with _open_text(source) as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            buffer += chunk

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise DataValidationError(f"Expected a top-level JSON array in {source}")
                    started = True
                    position += 1
                    continue

                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1

                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    position += 1
                    finished = True
                    break

                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as exc:
                    if eof:
                        raise DataValidationError(f"Invalid JSON array in {source}: {exc}") from exc
                    break
                yield value
                position = end

            if position:
                buffer = buffer[position:]
                position = 0
            if eof and not finished:
                raise DataValidationError(f"Unterminated JSON array in {source}")

        if buffer.strip():
            raise DataValidationError(f"Unexpected content after JSON array in {source}")


def _iter_json_lines(path: Path) -> Iterator[Any]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"Invalid JSON record in {path} at line {line_number}: {exc}"
                ) from exc


def _iter_delimited(path: Path, *, delimiter: str) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise DataValidationError(f"Graph table has no header: {path}")
        lookup = _column_lookup(list(reader.fieldnames), path)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise DataValidationError(
                    f"Graph {path} has extra values without a header at row {row_number}"
                )
            record = dict(row)
            for required in REQUIRED_GRAPH_COLUMNS:
                record[required] = row[lookup[required]]
            yield record


def _iter_xlsx(path: Path) -> Iterator[dict[str, Any]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        from tdg_bench.exceptions import OptionalDependencyError

        raise OptionalDependencyError(
            "Reading .xlsx graphs requires openpyxl. Install tdg-bench[gnn]."
        ) from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.iter_rows(values_only=True)
        try:
            header = list(next(rows))
        except StopIteration as exc:
            raise DataValidationError(f"Graph workbook is empty: {path}") from exc
        lookup = _column_lookup(header, path)
        indices = {
            required: header.index(lookup[required])
            for required in REQUIRED_GRAPH_COLUMNS
        }
        for row in rows:
            if not any(value is not None for value in row):
                continue
            yield {
                required: row[index] if index < len(row) else None
                for required, index in indices.items()
            }
    finally:
        workbook.close()


def iter_graph_records(path: str | Path) -> Iterator[Any]:
    """Yield graph records from JSON, JSONL, CSV, TSV or XLSX files.

    Text formats can optionally be gzip-compressed. Tabular headers are
    matched case-insensitively but must include subject, predicate and object.
    """
    source = Path(path)
    graph_format = _format_from_path(source)
    if graph_format == "json":
        yield from iter_json_array(source)
    elif graph_format in {"jsonl", "ndjson"}:
        yield from _iter_json_lines(source)
    elif graph_format == "csv":
        yield from _iter_delimited(source, delimiter=",")
    elif graph_format == "tsv":
        yield from _iter_delimited(source, delimiter="\t")
    else:
        yield from _iter_xlsx(source)


def iter_graph_triples(path: str | Path) -> Iterator[tuple[str, str, str]]:
    """Yield validated string triples using the same rules as evaluation."""
    source = Path(path)
    for index, record in enumerate(iter_graph_records(source)):
        if not isinstance(record, dict):
            raise DataValidationError(
                f"Graph record {index} in {source} is not an object"
            )
        values: list[str] = []
        for field in REQUIRED_GRAPH_COLUMNS:
            value = record.get(field)
            # Empty endpoints are retained for compatibility with the released
            # GT2KG graph, but absent and non-text values remain invalid.
            if not isinstance(value, str):
                raise DataValidationError(
                    f"Graph record {index} in {source} has invalid {field!r}"
                )
            values.append(value)
        yield values[0], values[1], values[2]

