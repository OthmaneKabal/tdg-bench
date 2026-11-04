"""Load every published Biomed variant and verify DBpedia rejects options."""

from __future__ import annotations

from pathlib import Path

from tdg_bench import TDGBench


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (None, "default"),
    (("raw",), "raw"),
    (("no_smnt",), "no_smnt"),
    (("no_freq_filter",), "no_freq_filter"),
    (("no_freq_filter", "no_smnt"), "no_freq_filter_no_smnt"),
)


def main() -> int:
    bench = TDGBench(PROJECT_ROOT / "benchmark_data")
    print(
        "Biomed options:",
        bench.available_preprocessing_options("biomed", "GT2KG_tdg"),
    )
    print(
        "DBpedia options:",
        bench.available_preprocessing_options("dbpedia", "GT2KG_tdg"),
    )

    for graph in ("GT2KG_tdg", "kggen"):
        print(f"Variants {graph}:", bench.available_variants("biomed", graph))
        for options, expected_variant in CASES:
            bundle = bench.get_data(
                dataset="biomed",
                graph=graph,
                options=options,
                split_seed=42,
                feature_dim=8,
                feature_seed=42,
                verify_graph=True,
            )
            if bundle.variant != expected_variant:
                raise AssertionError(
                    f"Expected {expected_variant}, got {bundle.variant}"
                )
            if bundle.coverage != 1.0:
                raise AssertionError(
                    f"Incomplete Gold Standard coverage: {bundle.coverage}"
                )
            print(
                f"OK biomed/{graph} options="
                f"{None if options is None else list(options)} -> "
                f"{bundle.variant} | nodes={bundle.data.num_nodes} "
                f"edges={bundle.data.num_edges}"
            )

    try:
        bench.get_data(
            dataset="dbpedia",
            graph="GT2KG_tdg",
            options=("raw",),
            split_seed=42,
            feature_dim=8,
        )
    except ValueError as error:
        print("OK DBpedia refused options:", error)
    else:
        raise AssertionError("DBpedia should reject preprocessing options")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
