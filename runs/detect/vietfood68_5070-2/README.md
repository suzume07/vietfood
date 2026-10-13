# VietFood68 — completed RTX 5070 training run

This directory is a byte-for-byte copy of the 40 original run files from `D:/vietfood/runs/detect/vietfood68_5070-2`. `upload_manifest.json` records file sizes and SHA-256 checksums. The source on D was left intact.

- Model: YOLO11s, 68 class IDs, image size 640, 150 epochs.
- Best validation mAP50-95: 0.84228, epoch 150.
- `weights/best.pt`: inference/evaluation checkpoint used by the report.
- `weights/last.pt`: final checkpoint; best/last have stripped optimizer state.
- `weights/epoch*.pt`: periodic checkpoints retained from training.
- `args.yaml`, `results.csv`, `results.png`: configuration and history.
- `resume-terminal.log`: partial log from earlier resumed sessions, not a complete transcript of all 150 epochs.

## Download large files

Checkpoints and the large terminal log use Git LFS. Install Git LFS, clone/check out `report/vietfood68-evaluation`, then run:

```powershell
git lfs install
git lfs pull
```

If a `.pt` file is only a short text pointer beginning with `version https://git-lfs.github.com/spec/v1`, run `git lfs pull` before loading it with Ultralytics.
