from __future__ import annotations

import pytest

from tdg_bench.datasets.variants import resolve_preprocessing_variant


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (("raw",), "raw"),
        (("no_smnt",), "no_smnt"),
        (("no_freq_filter",), "no_freq_filter"),
        (("no_smnt", "no_freq_filter"), "no_freq_filter_no_smnt"),
    ],
)
def test_biomed_preprocessing_options_resolve_to_variants(
    options: tuple[str, ...], expected: str
) -> None:
    assert resolve_preprocessing_variant("biomed", options=options) == expected


def test_dbpedia_has_no_preprocessing_options() -> None:
    with pytest.raises(ValueError, match="has no preprocessing options"):
        resolve_preprocessing_variant("dbpedia", options=("raw",))


def test_raw_is_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_preprocessing_variant("biomed", options=("raw", "no_smnt"))


def test_variant_and_options_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either variant or options"):
        resolve_preprocessing_variant(
            "biomed", variant="raw", options=("no_smnt",)
        )


def test_curated_umls_accepts_inverse_option() -> None:
    assert resolve_preprocessing_variant(
        "biomed", graph="cg", options=("with_inverse",)
    ) == "with_inverse"


def test_non_curated_biomed_rejects_inverse_option() -> None:
    with pytest.raises(ValueError, match="Unknown Biomed"):
        resolve_preprocessing_variant(
            "biomed", graph="GT2KG_tdg", options=("with_inverse",)
        )

