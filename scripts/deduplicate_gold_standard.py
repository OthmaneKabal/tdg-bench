"""Remove exact duplicate Gold Standard rows and remap every split safely."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook


PARTITIONS = ("train_idx", "val_idx", "test_idx")


def read_rows(path: Path) -> tuple[list[str], list[list[object]]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        values = list(sheet.iter_rows(values_only=True))
    finally:
        workbook.close()
    if not values:
        raise ValueError(f"Empty Gold Standard: {path}")
    header = [str(value) if value is not None else "" for value in values[0]]
    return header, [list(row) for row in values[1:]]


def deduplicate(
    header: list[str], rows: list[list[object]]
) -> tuple[list[list[object]], dict[int, int], list[dict[str, object]]]:
    try:
        term_column = header.index("term")
        label_column = header.index("label")
    except ValueError as exc:
        raise ValueError("Gold Standard requires 'term' and 'label' columns") from exc

    kept: list[list[object]] = []
    old_to_new: dict[int, int] = {}
    first_by_term: dict[str, tuple[int, object]] = {}
    removed: list[dict[str, object]] = []
    for old_index, row in enumerate(rows):
        term = str(row[term_column])
        label = row[label_column]
        previous = first_by_term.get(term)
        if previous is not None:
            kept_index, kept_label = previous
            if label != kept_label:
                raise ValueError(
                    f"Duplicate term {term!r} has conflicting labels: {kept_label!r}, {label!r}"
                )
            removed.append(
                {
                    "old_index": old_index,
                    "kept_old_index": next(
                        index
                        for index, mapped in old_to_new.items()
                        if mapped == kept_index
                    ),
                    "term": term,
                    "label": label,
                }
            )
            continue
        new_index = len(kept)
        first_by_term[term] = (new_index, label)
        old_to_new[old_index] = new_index
        kept.append(row)
    return kept, old_to_new, removed


def write_workbook(path: Path, header: list[str], rows: list[list[object]]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.xlsx")
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet()
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    workbook.save(temporary)
    os.replace(temporary, path)


def remap_splits(
    splits_dir: Path,
    old_to_new: dict[int, int],
    removed_indices: set[int],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split_path in sorted(splits_dir.glob("split_*.json")):
        split = json.loads(split_path.read_text(encoding="utf-8"))
        remapped: dict[str, list[int]] = {}
        for partition in PARTITIONS:
            values = [
                old_to_new[index]
                for index in split[partition]
                if index not in removed_indices
            ]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate remapped indices in {split_path}:{partition}")
            remapped[partition] = values
            np.save(
                splits_dir / f"{partition.removesuffix('_idx')}_idx_{split_path.stem.removeprefix('split_')}.npy",
                np.asarray(values, dtype=int),
            )
        temporary = split_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(remapped), encoding="utf-8")
        os.replace(temporary, split_path)
        summary[split_path.stem] = {
            "train_size": len(remapped["train_idx"]),
            "val_size": len(remapped["val_idx"]),
            "test_size": len(remapped["test_idx"]),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    gold_path = dataset_dir / "gold_standard" / "gold_standard.xlsx"
    splits_dir = dataset_dir / "splits"
    header, rows = read_rows(gold_path)
    kept, old_to_new, removed = deduplicate(header, rows)
    if not removed:
        print(f"No duplicate terms found in {gold_path}")
        return 0
    write_workbook(gold_path, header, kept)
    split_sizes = remap_splits(
        splits_dir,
        old_to_new,
        {int(item["old_index"]) for item in removed},
    )
    summary_path = splits_dir / "splits_summary.json"
    existing_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    for split_name, sizes in split_sizes.items():
        entry = existing_summary.setdefault(split_name, {})
        entry.update(sizes)
    summary_temporary = summary_path.with_suffix(".tmp")
    summary_temporary.write_text(
        json.dumps(existing_summary, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(summary_temporary, summary_path)
    report = {
        "original_rows": len(rows),
        "deduplicated_rows": len(kept),
        "removed": removed,
        "split_sizes": split_sizes,
    }
    report_path = dataset_dir / "gold_standard" / "deduplication_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
