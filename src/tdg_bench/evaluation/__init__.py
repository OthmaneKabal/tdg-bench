"""Evaluation API for TDGBench node-classification experiments."""

from tdg_bench.evaluation.runner import (
    BenchmarkResult,
    ClassificationMetrics,
    EpochMetrics,
    EvaluationResult,
    aggregate_results,
    evaluation_result_from_dict,
    evaluate_bundle,
)
from tdg_bench.evaluation.metrics import ClassMetrics, available_metric_names
from tdg_bench.evaluation.unified import (
    HiddenDimensionSummary,
    SplitMetricSummary,
    UnifiedEvaluationResult,
    UnifiedEvaluationSuite,
    VarianceSummary,
    run_unified_evaluation,
    run_unified_suite,
)

__all__ = [
    "BenchmarkResult",
    "ClassificationMetrics",
    "ClassMetrics",
    "EpochMetrics",
    "EvaluationResult",
    "aggregate_results",
    "evaluation_result_from_dict",
    "evaluate_bundle",
    "available_metric_names",
    "HiddenDimensionSummary",
    "SplitMetricSummary",
    "UnifiedEvaluationResult",
    "UnifiedEvaluationSuite",
    "VarianceSummary",
    "run_unified_evaluation",
    "run_unified_suite",
]
