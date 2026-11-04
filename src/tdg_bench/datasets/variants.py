"""Resolve the published preprocessing variants for benchmark graphs."""

from __future__ import annotations

from collections.abc import Iterable


BIOMED_PREPROCESSING_OPTIONS = ("raw", "no_smnt", "no_freq_filter")
CURATED_UMLS_PREPROCESSING_OPTIONS = (
    "raw",
    "no_smnt",
    "no_freq_filter",
    "with_inverse",
)
_BIOMED_ABLATIONS = frozenset({"no_smnt", "no_freq_filter"})
_CURATED_UMLS_ABLATIONS = frozenset({"no_smnt", "no_freq_filter", "with_inverse"})


def resolve_preprocessing_variant(
    dataset: str,
    *,
    graph: str | None = None,
    variant: str = "default",
    options: Iterable[str] | None = None,
) -> str:
    """Map public preprocessing options to one immutable manifest variant."""
    if options is None:
        return variant
    if isinstance(options, str):
        raise TypeError("options must be a list or tuple of option names, not a string")

    resolved_options = tuple(options)
    if not resolved_options:
        return variant
    if variant != "default":
        raise ValueError("Use either variant or options, not both")
    if dataset != "biomed":
        raise ValueError(
            f"Dataset {dataset!r} has no preprocessing options; "
            "use the default graph."
        )
    if any(
        not isinstance(option, str) or not option.strip()
        for option in resolved_options
    ):
        raise ValueError("Every preprocessing option must be a non-empty string")
    if len(set(resolved_options)) != len(resolved_options):
        raise ValueError("Preprocessing options must not contain duplicates")

    selected = frozenset(resolved_options)
    supported = (
        CURATED_UMLS_PREPROCESSING_OPTIONS
        if graph == "cg"
        else BIOMED_PREPROCESSING_OPTIONS
    )
    unknown = selected - set(supported)
    if unknown:
        raise ValueError(
            f"Unknown Biomed preprocessing option(s): {sorted(unknown)}. "
            f"Available options: {list(supported)}."
        )
    if "raw" in selected:
        if len(selected) != 1:
            raise ValueError("'raw' cannot be combined with other preprocessing options")
        return "raw"
    ablations = _CURATED_UMLS_ABLATIONS if graph == "cg" else _BIOMED_ABLATIONS
    if not selected <= ablations:
        raise ValueError(f"Unsupported Biomed preprocessing options: {sorted(selected)}")
    return "_".join(sorted(selected))

