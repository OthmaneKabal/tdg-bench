"""Exceptions raised by the public TDGBench API."""


class TDGBenchError(Exception):
    """Base exception for the library."""


class DatasetNotFoundError(TDGBenchError, ValueError):
    """Raised when a benchmark dataset is unknown."""


class GraphNotFoundError(TDGBenchError, ValueError):
    """Raised when a graph or graph variant is unavailable."""


class CorpusNotFoundError(TDGBenchError, ValueError):
    """Raised when a corpus artifact is unavailable."""


class InvalidManifestError(TDGBenchError, ValueError):
    """Raised when a dataset manifest does not follow the supported schema."""


class IntegrityError(TDGBenchError):
    """Raised when an asset does not match its manifest metadata."""


class DatasetDownloadError(TDGBenchError, RuntimeError):
    """Raised when a required Hub asset cannot be downloaded or found offline."""


class DataValidationError(TDGBenchError, ValueError):
    """Raised when graph, Gold Standard or split data are inconsistent."""


class OptionalDependencyError(TDGBenchError, ImportError):
    """Raised when an optional feature is used without its dependencies."""

