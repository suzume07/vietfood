"""Inspect and render the generated PDF for local quality checks."""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tools" / "pdf" / "python"))
import pymupdf
from PIL import Image, ImageDraw


def main():
    target = ROOT / "Bao_cao_VietFood68.pdf"
    previews = ROOT / ".cache" / "pdf" / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(target)
    texts = [page.get_text() for page in document]
    combined = "\n".join(texts)
    required = ["Nguyễn Ngọc Kiều Trinh", "95,34", "94,54", "97,23", "85,32", "class_0", "class_67", "TÀI LIỆU THAM KHẢO"]
    missing = [token for token in required if token not in combined]
    present_classes = set(map(int, re.findall(r"\bclass_(\d+)\b", combined)))
    missing_classes = [f"class_{i}" for i in range(68) if i not in present_classes]
    image_pages = [i+1 for i, page in enumerate(document) if page.get_images()]
    landmarks = {key: [i+1 for i, text in enumerate(texts) if key in text]
                 for key in ("MỤC LỤC", "PHỤ LỤC A.", "Hình 5a.", "Hình 5b.", "Hình 5c.", "Bảng 5.")}
    summary = {"pages": len(document), "bytes": target.stat().st_size, "missing_required_text": missing,
               "missing_class_ids": missing_classes, "unicode_replacement_characters": combined.count("\ufffd"),
               "image_pages": image_pages, "landmarks": landmarks,
               "blank_pages": [i+1 for i, text in enumerate(texts) if len(text.strip()) < 15 and i+1 not in image_pages],
               "rotated_pages": [i+1 for i, page in enumerate(document) if page.rotation],
               "bookmarks": len(document.get_toc()), "links": sum(len(page.get_links()) for page in document)}
    (ROOT / ".cache" / "pdf" / "quality-check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / ".cache" / "pdf" / "extracted-text.txt").write_text(combined, encoding="utf-8")
    width, height, columns = 250, 375, 5
    sheet = Image.new("RGB", (columns * width, math.ceil(len(document)/columns)*height), "#dce1e5")
    draw = ImageDraw.Draw(sheet)
    for i, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(.4, .4), alpha=False)
        preview = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        preview.thumbnail((width-12, height-26))
        x, y = (i % columns) * width, (i // columns) * height
        sheet.paste(preview, (x+(width-preview.width)//2, y+5))
        draw.text((x+10, y+height-18), f"PDF page {i+1}", fill="black")
    sheet.save(previews / "contact-sheet.png")
    selected = {0, 1, 2, 8, 14, 26, 27, 28, len(document)-1}
    for pages in landmarks.values():
        selected.update(p-1 for p in pages[-2:])
    for i in sorted(selected):
        document[i].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3), alpha=False).save(previews / f"page-{i+1:02d}.png")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    assert not missing and not missing_classes and not summary["blank_pages"]


if __name__ == "__main__":
    main()
