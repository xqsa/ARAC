"""提取关键文献中"优化阶段如何处理共享变量"的核心章节。"""
import re
from pathlib import Path

from pypdf import PdfReader

LIT = Path(r"E:\ARAC\references\lit-review")
TARGETS = {
    "Two-Phase CC.pdf": ["shared", "overlap", "two-phase", "phase"],
    "Investigating Overlapped Strategies to Solve Overlapping Problems in a Cooperative Co-evolutionary Framework.pdf": ["shared", "cooperation", "strategy"],
    "Contribution-Based_Cooperative_Co-Evolution_for_Nonseparable_Large-Scale_Problems_With_Overlapping_Subcomponents.pdf": ["shared variable", "assign", "contribution"],
    "Overlapping Cooperative Co-Evolution for Overlapping Large-Scale Global Optimization Problems.pdf": ["shared", "assignment", "resource"],
}

out = []
for name, kws in TARGETS.items():
    pdf = LIT / name
    reader = PdfReader(str(pdf))
    full = "\n".join((p.extract_text() or "") for p in reader.pages)
    full_norm = " ".join(full.split())
    out.append(f"\n{'#' * 100}\n# FILE: {name}  (pages: {len(reader.pages)}, chars: {len(full_norm)})\n{'#' * 100}")
    # 找出含关键词密度高的段落窗口
    sentences = re.split(r"(?<=[.!?])\s+", full_norm)
    hits = []
    for i, s in enumerate(sentences):
        low = s.lower()
        if any(k in low for k in kws) and len(s) > 60:
            hits.append(i)
    # 合并相邻命中，取窗口文本
    shown = 0
    last = -10
    for i in hits:
        if i - last < 3:
            continue
        last = i
        window = " ".join(sentences[max(0, i - 1): i + 2])
        out.append(f"\n[sent {i}] {window[:700]}")
        shown += 1
        if shown >= 25:
            break

Path(r"E:\ARAC\references\lit-review\shared_var_handling.txt").write_text(
    "\n".join(out), encoding="utf-8"
)
print("done")
