"""Coverage checks for user-provided graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tdg_bench.exceptions import DataValidationError, OptionalDependencyError
from tdg_bench.graphs.io import iter_graph_triples


@dataclass(frozen=True)
class GraphCoverageReport:
    """Common-node coverage of one graph against one benchmark domain."""

    dataset: str
    graph_name: str
    graph_path: Path
    graph_nodes: int
    graph_edges: int
    common_nodes_total: int
    common_nodes_covered: int
    missing_common_nodes: tuple[str, ...]

    @property
    def common_nodes_missing(self) -> int:
        return len(self.missing_common_nodes)

    @property
    def coverage(self) -> float:
        if self.common_nodes_total == 0:
            return 0.0
        return self.common_nodes_covered / self.common_nodes_total

    @property
    def is_complete(self) -> bool:
        return self.common_nodes_missing == 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["graph_path"] = str(self.graph_path)
        result["common_nodes_missing"] = self.common_nodes_missing
        result["coverage"] = self.coverage
        result["is_complete"] = self.is_complete
        return result


def _load_common_nodes(path: Path) -> tuple[str, ...]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - broken installation guard
        raise OptionalDependencyError(
            "Checking graph coverage requires openpyxl. "
            "Reinstall with: pip install --upgrade tdg-bench"
        ) from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.iter_rows(values_only=True)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise DataValidationError(f"Gold Standard is empty: {path}") from exc
        columns = {
            str(value).strip(): index
            for index, value in enumerate(header)
            if value is not None
        }
        if "term" not in columns:
            raise DataValidationError(
                f"Gold Standard {path} is missing the 'term' column"
            )
        terms: list[str] = []
        for row_number, row in enumerate(rows, start=2):
            value = row[columns["term"]]
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(f"Invalid term at {path}:{row_number}")
            terms.append(value.strip())
    finally:
        workbook.close()
    if not terms:
        raise DataValidationError(f"Gold Standard has no terms: {path}")
    return tuple(terms)


def check_graph_coverage(
    *,
    dataset: str,
    graph_path: str | Path,
    gold_standard_path: str | Path,
    graph_name: str | None = None,
) -> GraphCoverageReport:
    """Scan one graph and compare its nodes with the benchmark common nodes."""
    path = Path(graph_path).expanduser().resolve()
    if not path.is_file():
        raise DataValidationError(f"Graph file is missing: {path}")

    nodes: set[str] = set()
    edge_count = 0
    for subject, _predicate, obj in iter_graph_triples(path):
        nodes.add(subject)
        nodes.add(obj)
        edge_count += 1
    if edge_count == 0:
        raise DataValidationError(f"Graph contains no triples: {path}")

    common_nodes = _load_common_nodes(Path(gold_standard_path))
    missing = tuple(term for term in common_nodes if term not in nodes)
    return GraphCoverageReport(
        dataset=dataset,
        graph_name=graph_name or path.stem,
        graph_path=path,
        graph_nodes=len(nodes),
        graph_edges=edge_count,
        common_nodes_total=len(common_nodes),
        common_nodes_covered=len(common_nodes) - len(missing),
        missing_common_nodes=missing,
    )


def check_new_graph(
    graph_path: str | Path,
    common_nodes_path: str | Path,
    *,
    dataset: str = "custom",
    graph_name: str | None = None,
) -> GraphCoverageReport:
    """Public standalone coverage check that does not require ``TDGBench``."""
    return check_graph_coverage(
        dataset=dataset,
        graph_path=graph_path,
        gold_standard_path=common_nodes_path,
        graph_name=graph_name,
    )
