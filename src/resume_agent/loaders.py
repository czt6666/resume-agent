"""从本地文件读取简历正文（纯文本 / PDF）。"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class ResumeLoadError(ValueError):
    pass


def load_resume_text(path: str | Path) -> str:
    """根据后缀读取简历为纯文本。支持 .txt、.md、.pdf。"""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ResumeLoadError(f"文件不存在: {p}")

    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        raw = p.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        raw = _read_pdf(p)
    else:
        raise ResumeLoadError(f"不支持的格式 {suffix}，请使用 .txt / .md / .pdf")

    text = raw.strip()
    if not text:
        raise ResumeLoadError("文件内容为空或无法提取文本")
    return text


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)
