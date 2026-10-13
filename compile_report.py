"""Build a local Vietnamese LaTeX/PDF report from Markdown; never invokes Git."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / ".tools" / "pdf"
BUILD = ROOT / ".cache" / "pdf"


def tex_escape(text):
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=ROOT / "Bao_cao_VietFood68.md")
    args = parser.parse_args()
    source = args.source.resolve()
    text = source.read_text(encoding="utf-8")
    BUILD.mkdir(parents=True, exist_ok=True)
    pdf_dir = BUILD / "output"
    pdf_dir.mkdir(exist_ok=True)
    if not all((TOOLS / exe).is_file() for exe in ("pandoc.exe", "tectonic.exe")):
        raise SystemExit("Run .venv/Scripts/python.exe scripts/setup_pdf_tools.py first")
    title = re.findall(r"^# (.+)$", text, re.M)[1]
    front = text.split("## TÓM TẮT", 1)[0]
    metadata = re.findall(r"^\*\*(.+?):\*\*[ \t]+(.+?)[ \t]*$", front, re.M)
    metadata = [(k, v.replace("`", "")) for k, v in metadata]
    rows = "\n".join(tex_escape(k) + " & " + tex_escape(v) + r" \\[0.5em]" for k, v in metadata)
    cover = r"""\hypersetup{pageanchor=false}
\begin{titlepage}
\centering
{\large ĐẠI HỌC QUỐC GIA HÀ NỘI\par}
{\large\bfseries TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN\par}
\vspace{0.5cm}
{\normalsize KHOA TOÁN — CƠ — TIN HỌC\par}
\vspace{1.2cm}
\rule{0.8\linewidth}{0.7pt}\par
\vspace{0.8cm}
{\Large\bfseries BÁO CÁO DỰ ÁN\par}
\vspace{0.8cm}
{\LARGE\bfseries TITLE\par}
\vspace{0.8cm}
\rule{0.8\linewidth}{0.7pt}\par
\vfill
\begin{flushleft}
\begin{tabular}{@{}p{0.31\linewidth}p{0.66\linewidth}@{}}
ROWS
\end{tabular}
\end{flushleft}
\vfill
{\normalsize Hà Nội, tháng 10 năm 2026\par}
\end{titlepage}
\pagenumbering{arabic}
\hypersetup{pageanchor=true,pdftitle={TITLE},pdfauthor={AUTHOR}}
""".replace("TITLE", tex_escape(title)).replace("ROWS", rows)
    cover = cover.replace("AUTHOR", tex_escape(dict(metadata).get("Sinh viên thực hiện", "")))
    (BUILD / "cover.tex").write_text(cover, encoding="utf-8")
    tex_file = ROOT / (source.stem + ".tex")
    cmd = [str(TOOLS / "pandoc.exe"), str(source), "--from=markdown-implicit_figures", "--to=latex", "--standalone",
           "--lua-filter", str(ROOT / "scripts" / "pdf_report.lua"),
           "--include-in-header", str(ROOT / "scripts" / "pdf_header.tex"),
           "--include-before-body", str(BUILD / "cover.tex"), "--pdf-engine=xelatex",
           "-V", "documentclass=article", "-V", "fontsize=12pt", "-V", "papersize=a4",
           "-V", "geometry=left=2.5cm,right=2cm,top=2.2cm,bottom=2.2cm",
           "-V", "mainfont=Times New Roman", "-V", "sansfont=Arial", "-V", "monofont=Consolas",
           "-V", "colorlinks=true", "-M", "lang=vi", "--resource-path", str(ROOT),
           "--output", str(tex_file)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    environment = os.environ.copy()
    environment["TECTONIC_CACHE_DIR"] = str(TOOLS / "tex-cache")
    result = subprocess.run([str(TOOLS / "tectonic.exe"), str(tex_file), "--outdir", str(pdf_dir),
                             "--keep-logs", "--keep-intermediates", "--untrusted"],
                            cwd=ROOT, env=environment, text=True, encoding="utf-8", errors="replace", capture_output=True)
    (BUILD / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout[-6000:])
    print(result.stderr[-6000:])
    result.check_returncode()
    pdf = ROOT / (source.stem + ".pdf")
    shutil.copy2(pdf_dir / pdf.name, pdf)
    manifest = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "pdf": pdf.name, "latex": tex_file.name,
                "pandoc": subprocess.check_output([str(TOOLS / "pandoc.exe"), "--version"], text=True).splitlines()[0],
                "tectonic": subprocess.check_output([str(TOOLS / "tectonic.exe"), "--version"], text=True).strip(),
                "source_images_unchanged": True, "git_operations": False}
    (BUILD / "build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PDF written: {pdf} ({pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
