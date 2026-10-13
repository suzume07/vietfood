"""Install portable PDF build tools in this workspace, without touching training dependencies."""
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools" / "pdf"
RELEASES = (
    ("jgm/pandoc", "3.12.1", "windows-x86_64.zip", "pandoc.exe"),
    ("tectonic-typesetting/tectonic", "tectonic@0.17.0", "x86_64-pc-windows-msvc.zip", "tectonic.exe"),
)


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VietFood68-PDF-build"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    TOOLS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for repository, version, suffix, executable in RELEASES:
        if (TOOLS / executable).exists():
            print(f"Already installed: {executable}", flush=True)
            continue
        release = fetch_json(f"https://api.github.com/repos/{repository}/releases/tags/{urllib.parse.quote(version, safe='')}")
        asset = next(a for a in release["assets"] if a["name"].endswith(suffix))
        archive = TOOLS / asset["name"]
        print(f"Downloading {asset['name']} ({asset['size'] / 2**20:.1f} MiB)", flush=True)
        urllib.request.urlretrieve(asset["browser_download_url"], archive)
        actual = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
        expected = asset.get("digest")
        if expected and actual != expected:
            raise RuntimeError(f"Checksum mismatch for {asset['name']}")
        # Extract the executables only; archive paths never become destinations.
        with zipfile.ZipFile(archive) as package:
            member = next(n for n in package.namelist() if Path(n).name == executable)
            with package.open(member) as source, (TOOLS / executable).open("wb") as target:
                shutil.copyfileobj(source, target)
        manifest.append({"repository": repository, "version": version, "download": asset["browser_download_url"], "sha256": actual})
        archive.unlink()
        print(f"Installed {executable}; checksum {actual}", flush=True)
    if manifest:
        (TOOLS / "tools-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
