"""Create report figures and an image-level error audit from saved predictions."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import Rectangle
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ConfusionMatrix, box_iou

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "evaluation"
FIG = OUT / "figures"
RUN = Path("D:/vietfood/runs/detect/vietfood68_5070-2")
DATA = Path("D:/archive/dataset")


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "dataset_audit.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    rows = list(csv.DictReader((RUN / "results.csv").open()))
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key, name in (("precision", "Precision"), ("recall", "Recall"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")):
        axes[0, 0].plot(epochs, [float(r[f"metrics/{key}(B)"]) * 100 for r in rows], label=name)
    axes[0, 0].set(ylabel="%", title="Validation metrics — all 150 epochs")
    axes[0, 0].legend()
    for ax, loss in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["box", "cls", "dfl"]):
        for split in ("train", "val"):
            ax.plot(epochs, [float(r[f"{split}/{loss}_loss"]) for r in rows], label=split)
        ax.set(title=f"{loss.upper()} loss", ylabel="Loss")
        ax.legend()
    for ax in axes.flat:
        ax.set(xlabel="Epoch")
        ax.axvline(73, color="gray", linestyle=":", alpha=.6)
        ax.axvline(141, color="gray", linestyle="--", alpha=.6)
        ax.grid(alpha=.2)
    fig.savefig(FIG / "training_curves_150.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4), layout="constrained")
    x = np.arange(68)
    for offset, split in enumerate(("train", "valid", "test")):
        ax.bar(x + (offset - 1) * .28, audit["class_counts"][split]["instances"], width=.28, label=split)
    ax.set(yscale="log", xlabel="Class ID", ylabel="Labelled instances (log scale)", title="Class distribution")
    ax.legend()
    fig.savefig(FIG / "class_distribution.png", dpi=180)
    plt.close(fig)
    per_class = list(csv.DictReader((OUT / "test_per_class.csv").open()))
    fig, ax = plt.subplots(figsize=(13, 4), layout="constrained")
    ax.bar([int(r["class_id"]) for r in per_class], [100 * float(r["ap50_95"]) for r in per_class], color="#2581aa")
    ax.axhline(100 * summary["splits"]["test"]["map50_95"], color="#b44040", linestyle="--", label="Macro mean")
    ax.set(xlabel="Class ID", ylabel="AP50-95 (%)", ylim=(0, 100), title="Test AP50-95 by class")
    ax.legend()
    fig.savefig(FIG / "test_ap_by_class.png", dpi=180)
    plt.close(fig)
    predictions = json.loads((Path(summary["splits"]["test"]["save_dir"]) / "predictions.json").read_text())
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["file_name"]].append(row)
    conf = summary["splits"]["val"]["native_confidence_at_best_macro_f1"]
    matcher = SimpleNamespace(iouv=torch.tensor([.5]))
    cm = ConfusionMatrix(names={i: f"class_{i}" for i in range(68)})
    per_image, boxes_by_name, categories = [], {}, Counter()
    totals = Counter()
    for path in sorted((DATA / "images" / "test").glob("*.jpg")):
        with Image.open(path) as im:
            w, h = im.size
        gt_rows = [list(map(float, r.split())) for r in (DATA / "labels" / "test" / f"{path.stem}.txt").read_text().splitlines() if r.strip()]
        gt_cls = torch.tensor([r[0] for r in gt_rows])
        gt_boxes = torch.tensor([[(r[1] - r[3] / 2) * w, (r[2] - r[4] / 2) * h,
                                  (r[1] + r[3] / 2) * w, (r[2] + r[4] / 2) * h] for r in gt_rows]).reshape(-1, 4)
        pr_rows = [r for r in grouped[path.name] if r["score"] >= conf]
        pr_cls = torch.tensor([r["category_id"] - 1 for r in pr_rows])
        pr_conf = torch.tensor([r["score"] for r in pr_rows])
        pr_boxes = torch.tensor([[r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]] for r in pr_rows]).reshape(-1, 4)
        ious = box_iou(gt_boxes, pr_boxes)
        correct = BaseValidator.match_predictions(matcher, pr_cls, gt_cls, ious)[:, 0].numpy()
        tp, fp, fn = int(correct.sum()), len(pr_rows) - int(correct.sum()), len(gt_rows) - int(correct.sum())
        totals.update(tp=tp, fp=fp, fn=fn)
        for j in np.flatnonzero(~correct):
            if not len(gt_rows) or float(ious[:, j].max()) < .5:
                categories["no_gt_iou_ge_05"] += 1
            else:
                best = int(ious[:, j].argmax())
                categories["overlap_wrong_class" if gt_cls[best] != pr_cls[j] else "overlap_same_class_unmatched"] += 1
        cm.process_batch({"bboxes": pr_boxes, "cls": pr_cls, "conf": pr_conf},
                         {"bboxes": gt_boxes, "cls": gt_cls}, conf=conf, iou_thres=.5)
        per_image.append({"file_name": path.name, "gt": len(gt_rows), "predictions": len(pr_rows), "tp": tp, "fp": fp, "fn": fn})
        boxes_by_name[path.name] = (gt_boxes.numpy(), gt_cls.numpy(), pr_boxes.numpy(), pr_cls.numpy(), pr_conf.numpy())
    assert len(per_image) == summary["splits"]["test"]["images"], "Image count mismatch"
    errors = sorted(per_image, key=lambda r: (r["fp"] + r["fn"], r["fn"], r["fp"]), reverse=True)
    clean = [r for r in per_image if r["gt"] and not r["fp"] and not r["fn"]]
    sample = errors[:3] + sorted(clean, key=lambda r: r["gt"], reverse=True)[:1] + clean[len(clean)//2:len(clean)//2+1]
    empty = [r for r in per_image if r["gt"] == 0 and r["predictions"] == 0]
    if empty:
        sample += empty[:1]
    fig, axes = plt.subplots(len(sample), 2, figsize=(12, 3.7 * len(sample)), layout="constrained")
    for i, r in enumerate(sample):
        with Image.open(DATA / "images" / "test" / r["file_name"]) as im:
            pixels = np.asarray(im.convert("RGB"))
        gt_b, gt_c, pr_b, pr_c, pr_s = boxes_by_name[r["file_name"]]
        for col, (boxes, classes) in enumerate(((gt_b, gt_c), (pr_b, pr_c))):
            ax = axes[i, col]
            ax.imshow(pixels)
            for j, (b, c) in enumerate(zip(boxes, classes)):
                color = "#20db8a" if col == 0 else "#ffd239"
                ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, edgecolor=color, linewidth=1.4))
                label = f"ID {int(c)}" + (f" {pr_s[j]:.2f}" if col else "")
                ax.text(b[0], b[1], label, color="black", fontsize=7, bbox={"facecolor": color, "alpha": .85, "pad": 1})
            ax.set_title(("Ground truth" if col == 0 else f"Predictions — TP {r['tp']}, FP {r['fp']}, FN {r['fn']}") + f" | Sample {i+1}")
            ax.axis("off")
    fig.savefig(FIG / "test_error_examples.png", dpi=150)
    plt.close(fig)
    cm.plot(save_dir=FIG, normalize=True)
    (FIG / "confusion_matrix_normalized.png").replace(FIG / "test_operating_confusion_matrix.png")
    np.savetxt(OUT / "test_operating_confusion_matrix.csv", cm.matrix, delimiter=",", fmt="%.0f")
    with (OUT / "test_image_errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    offdiag = cm.matrix[:68, :68].copy()
    np.fill_diagonal(offdiag, 0)
    pairs = [(int(p), int(g), int(offdiag[p, g])) for p, g in zip(*np.where(offdiag > 0))]
    result = {"confidence": conf, "matching_iou": .5, "method": "Rematch predictions AFTER confidence filtering, original-image coordinates rounded by JSON exporter; class-aware greedy IoU matching.",
              "totals": dict(totals), "micro_precision": totals["tp"] / (totals["tp"]+totals["fp"]),
              "micro_recall": totals["tp"] / (totals["tp"]+totals["fn"]), "fp_categories": dict(categories),
              "images_with_errors": sum(r["fp"] + r["fn"] > 0 for r in per_image),
              "perfect_nonempty_images": len(clean), "empty_correct_images": len(empty),
              "examples": sample, "top_confusion_pairs": sorted(pairs, key=lambda r: r[2], reverse=True)[:10],
              "confusion_matrix_method": "Class-agnostic geometric matching; rows predicted, columns ground truth, last row/column background. TP may differ from class-aware AP matching."}
    (OUT / "error_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
