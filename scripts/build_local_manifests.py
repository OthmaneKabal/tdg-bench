"""Build deterministic manifests for datasets following the local layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SPLIT_FILE = re.compile(r"split_(\d+)\.json$")
TITLES = {
    "biomed": "TDGBench Biomedical",
    "dbpedia": "TDGBench DBpedia-174",
}


def asset(path: Path, dataset_root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.relative_to(dataset_root).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_manifest(dataset_root: Path) -> dict[str, object]:
    dataset_id = dataset_root.name
    gold_standard = dataset_root / "gold_standard" / "gold_standard.xlsx"
    if not gold_standard.is_file():
        raise FileNotFoundError(f"Missing Gold Standard: {gold_standard}")

    splits: dict[str, object] = {}
    for path in sorted((dataset_root / "splits").glob("split_*.json")):
        match = SPLIT_FILE.fullmatch(path.name)
        if match:
            splits[match.group(1)] = asset(path, dataset_root)
    if not splits:
        raise FileNotFoundError(f"No split JSON files found in {dataset_root / 'splits'}")

    graphs: dict[str, dict[str, object]] = {}
    graph_root = dataset_root / "graphs"
    for graph_directory in sorted(path for path in graph_root.iterdir() if path.is_dir()):
        variants = {
            path.stem: asset(path, dataset_root)
            for path in sorted(graph_directory.glob("*.json"))
        }
        if variants:
            graphs[graph_directory.name] = variants
    if not graphs:
        raise FileNotFoundError(f"No graph JSON files found in {graph_root}")

    corpus = {
        path.stem: asset(path, dataset_root)
        for path in sorted((dataset_root / "corpus").glob("*"))
        if path.is_file()
    }

    return {
        "schema_version": 1,
        "dataset": {
            "id": dataset_id,
            "title": TITLES.get(dataset_id, dataset_id),
            "version": "local-1",
        },
        "gold_standard": asset(gold_standard, dataset_root),
        "splits": splits,
        "graphs": graphs,
        "corpus": corpus,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    dataset_roots = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and path.name in TITLES
    )
    for dataset_root in dataset_roots:
        manifest = build_manifest(dataset_root)
        output = dataset_root / "manifest.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

