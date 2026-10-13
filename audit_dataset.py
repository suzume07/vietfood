"""Audit YOLO labels and exact cross-split duplicates without modifying data."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("D:/archive/dataset"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, summary, class_counts = {}, {}, {}
    for split in ("train", "valid", "test"):
        images = sorted(p for p in (args.dataset / "images" / split).iterdir() if p.suffix.lower() in EXTENSIONS)
        labels = {p.stem: p for p in (args.dataset / "labels" / split).glob("*.txt")}
        counts = np.zeros(68, dtype=int)
        image_counts = np.zeros(68, dtype=int)
        missing, invalid, empty = [], [], 0
        for image in images:
            label = labels.get(image.stem)
            if label is None:
                missing.append(image.name)
                continue
            rows = label.read_text(encoding="utf-8-sig").splitlines()
            present = set()
            if not any(row.strip() for row in rows):
                empty += 1
            for line_number, row in enumerate(rows, 1):
                if not row.strip():
                    continue
                try:
                    values = np.asarray([float(x) for x in row.split()])
                    if (len(values) != 5 or not np.isfinite(values).all() or
                            values[0] != int(values[0]) or not 0 <= values[0] < 68 or
                            not ((values[1:] >= 0) & (values[1:] <= 1)).all() or
                            not (values[3:] > 0).all()):
                        raise ValueError("invalid YOLO bbox")
                    c = int(values[0])
                    counts[c] += 1
                    present.add(c)
                except ValueError:
                    invalid.append({"file": label.name, "line": line_number})
            for c in present:
                image_counts[c] += 1
        records[split] = images
        class_counts[split] = {"instances": counts.tolist(), "images": image_counts.tolist()}
        summary[split] = {"images": len(images), "labels": len(labels), "empty_labels": empty,
                          "instances": int(counts.sum()), "classes_present": int((counts > 0).sum()),
                          "min_instances_per_class": int(counts.min()), "max_instances_per_class": int(counts.max()),
                          "missing_labels": missing, "invalid_rows": invalid,
                          "orphan_labels": sorted(set(labels) - {p.stem for p in images}),
                          "total_image_bytes": sum(p.stat().st_size for p in images)}
        print(f"LABEL AUDIT {split}: {summary[split]}", flush=True)
    # Hash all holdout images, then only train files of candidate byte sizes.
    holdout = records["valid"] + records["test"]
    sizes = {p.stat().st_size for p in holdout}
    candidates = holdout + [p for p in records["train"] if p.stat().st_size in sizes]
    hashes = defaultdict(list)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, sha in zip(candidates, pool.map(digest, candidates)):
            hashes[sha].append(path)
    duplicates = []
    contaminated = {split: set() for split in ("valid", "test")}
    for sha, paths in hashes.items():
        splits = {p.parent.name for p in paths}
        if len(splits) > 1:
            duplicates.append({"sha256": sha, "files": [str(p) for p in paths]})
            if "train" in splits:
                for p in paths:
                    if p.parent.name in contaminated:
                        contaminated[p.parent.name].add(p.name)
    # Roboflow base-name collisions are hints, not proof of common source.
    bases = {s: Counter(p.name.split(".rf.")[0] for p in paths) for s, paths in records.items()}
    overlaps = {}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        common = sorted(set(bases[a]) & set(bases[b]))
        overlaps[f"{a}_{b}"] = {"distinct_base_names": len(common), "examples": common[:20],
                                 "images_in_second_split": sum(bases[b][name] for name in common)}
    output = {"dataset": str(args.dataset), "splits": summary, "class_counts": class_counts,
              "sha256_files_checked": len(candidates), "cross_split_exact_duplicate_groups": len(duplicates),
              "holdout_images_exactly_in_train": {s: len(v) for s, v in contaminated.items()},
              "duplicate_groups": duplicates, "base_name_overlap": overlaps,
              "caveat": "Base-name collisions do not prove leakage. SHA256 detects byte-identical files only; transformed near duplicates and source-video overlap require a separate audit."}
    (args.output / "dataset_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "dataset_class_counts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "train_instances", "valid_instances", "test_instances", "train_images", "valid_images", "test_images"])
        for c in range(68):
            writer.writerow([c] + [class_counts[s][metric][c] for metric in ("instances", "images") for s in ("train", "valid", "test")])
    print(f"DUPLICATE AUDIT: groups={len(duplicates)}, holdout-in-train={output['holdout_images_exactly_in_train']}", flush=True)


if __name__ == "__main__":
    main()
