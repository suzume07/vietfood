"""Warm, single-image PyTorch inference benchmark on a fixed 100-image sample."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "evaluation"


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    images = sorted(Path("D:/archive/dataset/images/test").glob("*.jpg"))
    selected = [images[i] for i in np.linspace(0, len(images)-1, 100, dtype=int)]
    model = YOLO(summary["model_path"])
    kwargs = dict(imgsz=640, batch=1, device=0, quantize=16, conf=summary["splits"]["val"]["native_confidence_at_best_macro_f1"],
                  iou=.7, max_det=300, verbose=False, save=False, augment=False, rect=False)
    for image in selected[:3]:
        model.predict(str(image), **kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # A Python list is loaded as one image-list batch by Ultralytics regardless
    # of the batch argument. Separate calls are required to measure batch=1.
    speeds = []
    for image in selected:
        result = model.predict(str(image), **kwargs)[0]
        speeds.append(dict(result.speed))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    record = {"images": len(speeds), "sample": [p.name for p in selected],
              "protocol": kwargs, "warmup_images": 3, "wall_seconds": elapsed,
              "wall_ms_per_image": elapsed * 1000 / len(speeds),
              "wall_images_per_second": len(speeds) / elapsed,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "speed_ms": {key: {"mean": float(np.mean([r[key] for r in speeds])),
                                 "median": float(np.median([r[key] for r in speeds])),
                                 "p95": float(np.percentile([r[key] for r in speeds], 95))}
                           for key in ("preprocess", "inference", "postprocess")},
              "caveat": "One warm local run, 100 separate predict calls, batch=1. Wall timing includes file loading and Python/API overhead; excludes model loading, warmup, UI, networking and result rendering."}
    (OUT / "benchmark.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "sample"}, indent=2))


if __name__ == "__main__":
    main()
