"""Implementation manifest hashing for gate experiments.

每个实验 cell 必须携带 ``implementation_manifest_hash``——对调度器、
state、协议与配置文件内容的联合哈希。混合版本收据由此被结构性
排除：任何被覆盖文件的内容变化都会改变 manifest，实验脚本在
启动时比对失败即响亮退出。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_MANIFEST_ALGORITHM = "sha256"


def implementation_manifest_hash(repo_root: Path, files: list[Path]) -> str:
    """Compute a deterministic content hash over the given files.

    ``files`` are paths relative to ``repo_root`` (absolute paths also
    accepted if they live under it). Files are sorted by their relative
    POSIX path, each contributing ``<relpath>:<sha256(content)>``; the
    manifest hash is the hex digest of the newline-joined contribution
    lines. Missing files raise ``FileNotFoundError`` loudly.
    """

    root = Path(repo_root).resolve()
    contributions: list[tuple[str, str]] = []
    for path in files:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = root / resolved
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest file missing: {resolved}")
        content_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
        rel = resolved.relative_to(root).as_posix()
        contributions.append((rel, content_hash))
    contributions.sort()
    lines = [f"{rel}:{digest}" for rel, digest in contributions]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return f"{_MANIFEST_ALGORITHM}:{digest}"
