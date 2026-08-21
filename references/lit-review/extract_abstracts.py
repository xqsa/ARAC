"""批量提取文献首页摘要文本，用于快速筛选相关机制。"""
import sys
from pathlib import Path

from pypdf import PdfReader

LIT = Path(r"E:\ARAC\references\lit-review")
OUT = Path(r"E:\ARAC\references\lit-review\abstracts.txt")

chunks: list[str] = []
for pdf in sorted(LIT.glob("*.pdf")):
    try:
        reader = PdfReader(str(pdf))
        text = ""
        for page in reader.pages[:2]:
            text += (page.extract_text() or "") + "\n"
        text = " ".join(text.split())
        chunks.append(f"\n{'=' * 100}\nFILE: {pdf.name}\n{'=' * 100}\n{text[:2600]}\n")
    except Exception as exc:  # noqa: BLE001
        chunks.append(f"\n{'=' * 100}\nFILE: {pdf.name}\nERROR: {exc}\n")

OUT.write_text("".join(chunks), encoding="utf-8")
print(f"wrote {OUT}, {len(chunks)} files")
