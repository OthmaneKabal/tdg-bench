"""Prepare a clean, checksummed TDGBench payload for Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "benchmark_data"
ASSETS_ROOT = PROJECT_ROOT / "hf_dataset_assets"
DATASETS = ("biomed", "dbpedia")
RELEASE_VERSION = "2.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_declared_asset(data_root: Path, value: dict[str, Any]) -> Path:
    path = (data_root / value["path"]).resolve()
    if not path.is_relative_to(data_root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Missing or invalid manifest asset: {path}")
    if path.stat().st_size != value["size"]:
        raise ValueError(f"Size mismatch for {path}")
    if sha256(path) != value["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def declared_assets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    assets = [manifest["gold_standard"], *manifest["splits"].values()]
    for variants in manifest["graphs"].values():
        assets.extend(variants.values())
    assets.extend(manifest.get("corpus", {}).values())
    return assets


def prepare_dataset(data_root: Path, output_root: Path, dataset: str) -> None:
    source_root = data_root / dataset
    manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for asset in declared_assets(manifest):
        source = verify_declared_asset(source_root, asset)
        copy_file(source, output_root / dataset / asset["path"])

    summary = source_root / "splits" / "splits_summary.json"
    if summary.is_file():
        copy_file(summary, output_root / dataset / "splits" / summary.name)

    manifest["dataset"]["version"] = RELEASE_VERSION
    destination = output_root / dataset / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / f"hf_dataset_v{RELEASE_VERSION}",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Choose a new --output path."
        )
    output_root.mkdir(parents=True)

    try:
        for name in ("README.md", "LICENSE_DATA.md", "THIRD_PARTY_NOTICES.md"):
            copy_file(ASSETS_ROOT / name, output_root / name)
        for dataset in DATASETS:
            prepare_dataset(data_root, output_root, dataset)

        root_manifest = {
            "schema_version": 1,
            "release": RELEASE_VERSION,
            "datasets": {
                dataset: metadata(
                    output_root / dataset / "manifest.json", output_root
                )
                for dataset in DATASETS
            },
        }
        (output_root / "manifest.json").write_text(
            json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_root)
        raise

    files = [path for path in output_root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Prepared {len(files)} files in {output_root}")
    print(f"Payload size: {total_bytes / (1024 ** 3):.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
