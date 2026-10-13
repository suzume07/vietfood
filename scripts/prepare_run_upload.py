"""Copy one completed training run into the repo and verify every source file."""
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("D:/vietfood/runs/detect/vietfood68_5070-2")
DESTINATION = ROOT / "runs" / "detect" / SOURCE.name


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source = SOURCE.resolve(strict=True)
    destination = DESTINATION.resolve()
    if not destination.is_relative_to(ROOT.resolve()) or destination == source:
        raise RuntimeError("Destination must stay inside the workspace and differ from source")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(source) for p in files):
        raise RuntimeError("Source contains links pointing outside the named run")
    total = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(ROOT).free < total * 2 + 2 * 1024**3:
        raise RuntimeError("Insufficient free disk space for the working copy and LFS objects")
    entries = []
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        before = path.stat()
        checksum = sha256(path)
        if target.exists() and sha256(target) != checksum:
            raise RuntimeError(f"Refusing to overwrite a different existing file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        if sha256(target) != checksum:
            raise RuntimeError(f"Copy checksum mismatch: {relative}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during copy: {relative}")
        if path.suffix in {".yaml", ".csv", ".log"}:
            data = path.read_bytes()
            encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
            content = data.decode(encoding, errors="replace")
            patterns = (r"gh[pousr]_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9_-]{24,}",
                        r"AKIA[A-Z0-9]{16}", r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")
            if any(re.search(pattern, content) for pattern in patterns):
                raise RuntimeError(f"Credential pattern found; review file before uploading: {relative}")
        entries.append({"path": relative.as_posix(), "bytes": before.st_size, "sha256": checksum,
                        "git_lfs": path.suffix == ".pt" or path.name == "resume-terminal.log"})
        print(f"Verified {relative.as_posix()} ({before.st_size:,} bytes)", flush=True)
    manifest = {"source": str(source), "destination": destination.relative_to(ROOT).as_posix(),
                "prepared_utc": datetime.now(timezone.utc).isoformat(),
                "source_file_count": len(entries), "source_bytes": total,
                "lfs_file_count": sum(e["git_lfs"] for e in entries),
                "lfs_bytes": sum(e["bytes"] for e in entries if e["git_lfs"]), "files": entries}
    (destination / "upload_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "README.md").write_text(
        "# VietFood68 — completed RTX 5070 training run\n\n"
        "This directory is a byte-for-byte copy of the 40 original run files from "
        "`D:/vietfood/runs/detect/vietfood68_5070-2`. `upload_manifest.json` records "
        "file sizes and SHA-256 checksums. The source on D was left intact.\n\n"
        "- Model: YOLO11s, 68 class IDs, image size 640, 150 epochs.\n"
        "- Best validation mAP50-95: 0.84228, epoch 150.\n"
        "- `weights/best.pt`: inference/evaluation checkpoint used by the report.\n"
        "- `weights/last.pt`: final checkpoint; best/last have stripped optimizer state.\n"
        "- `weights/epoch*.pt`: periodic checkpoints retained from training.\n"
        "- `args.yaml`, `results.csv`, `results.png`: configuration and history.\n"
        "- `resume-terminal.log`: partial log from earlier resumed sessions, not "
        "a complete transcript of all 150 epochs.\n\n"
        "## Download large files\n\n"
        "Checkpoints and the large terminal log use Git LFS. Install Git LFS, "
        "clone/check out `report/vietfood68-evaluation`, then run:\n\n"
        "```powershell\ngit lfs install\ngit lfs pull\n```\n\n"
        "If a `.pt` file is only a short text pointer beginning with "
        "`version https://git-lfs.github.com/spec/v1`, run `git lfs pull` before "
        "loading it with Ultralytics.\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
