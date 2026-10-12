"""YOLO detection training baseline for RTX 5070 desktop."""
import argparse
import multiprocessing
import os
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path("D:/vietfood/runs/detect") if os.name == "nt" and Path("D:/").is_dir() else ROOT / "runs" / "detect"


def batch_value(value):
    n = float(value)
    if n == -1 or (n >= 1 and n.is_integer()):
        return int(n)
    if 0 < n < 1:
        return n
    raise argparse.ArgumentTypeError("Use -1, a positive integer, or a fraction in (0, 1)")


def enable_benchmark(trainer):
    # Called after AutoBatch profiling: enabling this earlier disables AutoBatch.
    import torch
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=batch_value, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--prefetch", type=int, default=2, help="Batches queued per train worker (validation has no workers)")
    parser.add_argument("--max-gpu", action="store_true", help="Target 90%% VRAM; keep Windows loader workers conservative")
    parser.add_argument("--cache", choices=("off", "disk", "ram"), default="off")
    parser.add_argument("--name", default="vietfood68_5070")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output directory (Windows default: D:/vietfood/runs/detect)")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One epoch on 5%% of train data")
    args = parser.parse_args()
    if args.batch is None and not args.resume:
        args.batch = 0.90 if args.max_gpu else 0.65
    if args.workers is None:
        # More workers multiply dataset state, image buffers and prefetched batches.
        # GPU VRAM capacity does not imply enough host RAM for 8 Windows processes.
        worker_limit = 2 if os.name == "nt" else (8 if args.max_gpu else 4)
        args.workers = min(worker_limit, max(0, (os.cpu_count() or 2) - 2))
    if args.epochs < 1 or args.imgsz < 32 or args.imgsz % 32 or args.workers < 0 or args.prefetch < 1:
        parser.error("epochs >= 1, imgsz a positive multiple of 32, workers >= 0, prefetch >= 1 required")
    if args.resume and args.smoke:
        parser.error("--resume and --smoke cannot be combined")

    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from check_gpu import check_gpu
    from memory_trainer import DesktopDetectionTrainer
    trainer_factory = partial(DesktopDetectionTrainer, loader_prefetch=args.prefetch)

    check_gpu()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        checkpoint = args.resume.resolve()
        model = YOLO(str(checkpoint))
        model.add_callback("on_pretrain_routine_end", enable_benchmark)
        # Resume permits save_dir overrides, but ignores project/name. Follow the
        # checkpoint's current run directory so relocated runs stay on that drive.
        resume_options = dict(trainer=trainer_factory, resume=True, device=0,
                              workers=args.workers,
                              cache=False if args.cache == "off" else args.cache,
                              save_dir=str(checkpoint.parent.parent))
        # An omitted batch preserves the checkpoint setting. Explicit --batch
        # is supported by Ultralytics resume and must actually reach the trainer.
        if args.batch is not None:
            resume_options["batch"] = args.batch
        model.train(**resume_options)
        return
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    dataset = check_det_dataset(str(args.data.resolve()), autodownload=False)
    if any(str(name).startswith("class_") for name in dataset["names"].values()):
        print("WARNING: placeholder class names; verify dataset label mapping.")
    options = dict(
        data=str(args.data.resolve()), device=0, epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        cache=False if args.cache == "off" else args.cache,
        amp=True, deterministic=False, seed=42, optimizer="auto",
        cos_lr=True, patience=30, close_mosaic=10,
        project=str(args.project.resolve()), name=args.name,
        save=True, save_period=10, val=True, plots=True,
    )
    if args.smoke:
        options.update(epochs=1, fraction=0.05, batch=2, workers=0,
                       cache=False, close_mosaic=0, name=args.name + "_smoke")
    print("Training options:", options)
    model = YOLO(args.model)
    model.add_callback("on_pretrain_routine_end", enable_benchmark)
    model.train(trainer=trainer_factory, **options)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
