"""长期记忆：按 user_id 持久化简历解析档案（无 LangChain 依赖）。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from resume_agent.schemas import ParsedResume

LONG_TERM_FILENAME = "long_term.json"


def default_data_dir() -> Path:
    raw = os.environ.get("RESUME_AGENT_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".resume_agent"


def sanitize_user_id(user_id: str) -> str:
    """仅允许安全字符，防止路径穿越。"""
    s = user_id.strip()
    if not s:
        return "default"
    if s in {".", ".."}:
        raise ValueError("user_id 不能使用 . 或 ..")
    if not re.fullmatch(r"[\w.\-]{1,64}", s):
        raise ValueError(
            "user_id 仅允许字母数字、下划线、点、连字符，长度 1～64"
        )
    return s


def user_storage_dir(user_id: str) -> Path:
    sid = sanitize_user_id(user_id)
    return default_data_dir() / sid


@dataclass(frozen=True)
class LongTermRecord:
    """磁盘上的长期记忆结构。"""

    version: int
    updated_at: str
    resume_fingerprint: str
    parsed_resume: dict[str, Any]

    @classmethod
    def from_parsed(cls, raw_resume_text: str, parsed: ParsedResume) -> LongTermRecord:
        fp = sha256(raw_resume_text.encode("utf-8")).hexdigest()
        return cls(
            version=1,
            updated_at=datetime.now(timezone.utc).isoformat(),
            resume_fingerprint=fp,
            parsed_resume=parsed.model_dump(),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "resume_fingerprint": self.resume_fingerprint,
            "parsed_resume": self.parsed_resume,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> LongTermRecord:
        return cls(
            version=int(data.get("version", 1)),
            updated_at=str(data.get("updated_at", "")),
            resume_fingerprint=str(data.get("resume_fingerprint", "")),
            parsed_resume=dict(data.get("parsed_resume") or {}),
        )


def long_term_prompt_block(record: LongTermRecord) -> str:
    """注入到用户消息前的长期记忆说明（不含简历原文）。"""
    body = json.dumps(record.parsed_resume, ensure_ascii=False, indent=2)
    return (
        "【用户长期记忆·简历结构化档案（最近一次上传并解析）】\n"
        f"{body}\n"
        "说明：姓名、意向岗位、技能等以本档案为准；若用户口述与档案冲突，以当轮表述为准并提醒其更新简历文件后重新上传。\n"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_long_term(user_id: str) -> LongTermRecord | None:
    path = user_storage_dir(user_id) / LONG_TERM_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "parsed_resume" not in data:
        return None
    try:
        return LongTermRecord.from_json_dict(data)
    except (TypeError, ValueError):
        return None


def save_long_term(user_id: str, record: LongTermRecord) -> None:
    path = user_storage_dir(user_id) / LONG_TERM_FILENAME
    _atomic_write_json(path, record.to_json_dict())
