"""Reproducible holdout evaluation; choose operating confidence on validation only."""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import smooth

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class AuditValidator(DetectionValidator):
    instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        AuditValidator.instance = self

    def get_stats(self):
        # Stock validator clears these after processing. Retain matching results
        # to count TP/FP/FN at a confidence frozen before test evaluation.
        self.raw_stats = {k: np.concatenate(v, axis=0) for k, v in self.metrics.stats.items()}
        return super().get_stats()


def operating_metrics(stats, threshold, nc):
    keep = stats["conf"] >= threshold
    predicted = stats["pred_cls"][keep].astype(int)
    true_positive = stats["tp"][keep, 0].astype(bool)
    tp = np.bincount(predicted[true_positive], minlength=nc)
    predicted_count = np.bincount(predicted, minlength=nc)
    target_count = np.bincount(stats["target_cls"].astype(int), minlength=nc)
    p = np.divide(tp, predicted_count, out=np.zeros(nc, dtype=float), where=predicted_count > 0)
    r = np.divide(tp, target_count, out=np.zeros(nc, dtype=float), where=target_count > 0)
    present = target_count > 0
    total_tp, total_pred, total_gt = int(tp.sum()), int(predicted_count.sum()), int(target_count.sum())
    return {
        "confidence": float(threshold), "matching_iou": 0.5,
        "tp": total_tp, "fp": total_pred - total_tp, "fn": total_gt - total_tp,
        "micro_precision": total_tp / max(total_pred, 1),
        "micro_recall": total_tp / max(total_gt, 1),
        "macro_precision": float(p[present].mean()), "macro_recall": float(r[present].mean()),
        "per_class_tp": tp.tolist(), "per_class_fp": (predicted_count - tp).tolist(),
        "per_class_fn": (target_count - tp).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("D:/vietfood/runs/detect/vietfood68_5070-2/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument("--project", type=Path, default=Path("D:/vietfood/evaluation"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    model = YOLO(str(args.model))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "checkpoint_bytes": args.model.stat().st_size,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "architecture_yaml": model.model.yaml,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0),
                        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
                        "cpu_physical": psutil.cpu_count(logical=False), "cpu_logical": psutil.cpu_count(),
                        "ram_bytes": psutil.virtual_memory().total},
        "protocol": {"imgsz": 640, "batch": args.batch, "workers": 0, "device": 0,
                     "half": True, "conf_floor": 0.001, "nms_iou": 0.7,
                     "max_det": 300, "augment": False, "rect": True},
        "splits": {},
    }
    summary["environment"]["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    (args.output / "environment-freeze.txt").write_text(
        subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    for filename, output_name in (("results.csv", "training_results.csv"), ("args.yaml", "training_args.yaml")):
        source = args.model.parent.parent / filename
        if source.is_file():
            shutil.copy2(source, args.output / output_name)
    frozen_confidence = None
    for split in ("val", "test"):
        metrics = model.val(validator=AuditValidator, data=str(args.data), split=split,
                            imgsz=640, batch=args.batch, workers=0, device=0, half=True,
                            conf=0.001, iou=0.7, max_det=300, augment=False, rect=True,
                            plots=True, save_json=True, project=str(args.project),
                            name=f"best_{split}", exist_ok=False, verbose=False)
        validator = AuditValidator.instance
        b = metrics.box
        selected_conf = float(b.px[smooth(b.f1_curve.mean(0), 0.1).argmax()])
        if split == "val":
            frozen_confidence = selected_conf
        stats = validator.raw_stats
        np.savez_compressed(args.output / f"{split}_matching_stats.npz", **stats)
        np.savez_compressed(args.output / f"{split}_curves.npz", confidence=b.px,
                            precision=b.p_curve, recall=b.r_curve, f1=b.f1_curve)
        record = {
            "precision": float(b.mp), "recall": float(b.mr),
            "map50": float(b.map50), "map50_95": float(b.map),
            "native_confidence_at_best_macro_f1": selected_conf,
            "images": int(validator.seen), "instances": int(metrics.nt_per_class.sum()),
            "classes_present": len(b.ap_class_index),
            "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
            "save_dir": str(validator.save_dir),
            "operating_at_val_confidence": operating_metrics(stats, frozen_confidence, len(model.names)),
            "operating_at_025": operating_metrics(stats, 0.25, len(model.names)),
        }
        summary["splits"][split] = record
        with (args.output / f"{split}_per_class.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_id", "name", "images", "instances", "precision_native", "recall_native", "ap50", "ap50_95", "tp_at_val_conf", "fp_at_val_conf", "fn_at_val_conf"])
            op = record["operating_at_val_confidence"]
            for j, class_id in enumerate(b.ap_class_index):
                c = int(class_id)
                writer.writerow([c, model.names[c], int(metrics.nt_per_image[c]), int(metrics.nt_per_class[c]),
                                 float(b.p[j]), float(b.r[j]), float(b.ap50[j]), float(b.ap[j]),
                                 op["per_class_tp"][c], op["per_class_fp"][c], op["per_class_fn"][c]])
        np.savetxt(args.output / f"{split}_confusion_matrix.csv", validator.confusion_matrix.matrix,
                   delimiter=",", fmt="%.0f")
        for source in validator.save_dir.glob("*"):
            if source.suffix.lower() in (".png", ".jpg"):
                shutil.copy2(source, figures / f"{split}_{source.name}")
        write_json(args.output / "summary.json", summary)
        print(f"AUDIT {split}: {json.dumps(record, ensure_ascii=True)}", flush=True)
    (args.output / "environment-freeze-final.txt").write_text(
        subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(f"Report artifacts: {args.output}", flush=True)


if __name__ == "__main__":
    main()
