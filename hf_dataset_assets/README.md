---
pretty_name: TDGBench
license: other
license_name: tdgbench-file-level-licenses
license_link: https://huggingface.co/datasets/othmanekabal/TDGBench/blob/main/LICENSE_DATA.md
task_categories:
  - graph-ml
language:
  - en
tags:
  - knowledge-graph
  - graph-neural-networks
  - text-driven-graphs
  - benchmark
---

# TDGBench

TDGBench provides immutable graph artifacts, Gold Standards and ten fixed
train/validation/test splits for reproducible evaluation of text-driven
knowledge graphs with graph neural networks.

## Contents

```text
biomed/
  graphs/GT2KG_tdg/{default,raw,no_smnt,no_freq_filter,no_freq_filter_no_smnt}.json
  graphs/kggen/{default,raw,no_smnt,no_freq_filter,no_freq_filter_no_smnt}.json
  corpus/medmentions_mapped.json
  gold_standard/gold_standard.xlsx
  splits/split_<seed>.json
  manifest.json
dbpedia/
  graphs/GT2KG_tdg/default.json
  graphs/cg/default.json
  corpus/dbpedia_174_abstracts.json
  gold_standard/gold_standard.xlsx
  splits/split_<seed>.json
  manifest.json
manifest.json
```

The predefined split seeds are `42`, `123`, `456`, `789`, `2024`, `2025`,
`2026`, `2027`, `2028` and `2029`. Every manifest records the byte size and
SHA-256 digest of each benchmark asset.

## Biomed preprocessing options

- `raw`: no preprocessing;
- `no_smnt`: without semantic-type integration;
- `no_freq_filter`: without relation-frequency filtering;
- `no_freq_filter_no_smnt`: both ablations together;
- `default`: the fully processed published graph.

DBpedia has no preprocessing variants.

## Corpora

- `biomed/corpus/medmentions_mapped.json`: the mapped MedMentions corpus used
  to construct the biomedical TDGs;
- `dbpedia/corpus/dbpedia_174_abstracts.json`: the DBpedia-174 abstract corpus
  used to construct the DBpedia TDG.

The corpus assets are declared in their domain manifest and protected by
byte-size and SHA-256 checks.

## Use with tdg-bench

The companion Python library is maintained at
[OthmaneKabal/tdg-bench](https://github.com/OthmaneKabal/tdg-bench). It loads
these artifacts, validates their checksums, applies the fixed splits and runs
the unified graph-construction, no-graph MLP and GNN evaluation protocols.

```bash
pip install tdg-bench
```

## Important UMLS restriction

The UMLS-NCI graph and its generated variants are **not included**. They must
be reconstructed locally by users holding a valid UMLS licence. See
`LICENSE_DATA.md` and `THIRD_PARTY_NOTICES.md` for file-level terms.

## Release

This directory is the data payload for TDGBench `v2.0.0`, extending the
Biomedical benchmark from `v1.0.0` with the DBpedia-174 domain. Consumers
should pin the Hugging Face revision tag rather than an unversioned branch.
