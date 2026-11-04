"""Public interface for TDGBench."""

from tdg_bench.benchmark import TDGBench
from tdg_bench.datasets.manifest import GraphArtifact
from tdg_bench.graphs.coverage import GraphCoverageReport, check_new_graph
__all__ = [
    "GraphArtifact",
    "GraphCoverageReport",
    "TDGBench",
    "check_new_graph",
]
__version__ = "2.0.0rc5"

