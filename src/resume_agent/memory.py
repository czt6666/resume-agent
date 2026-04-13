"""按 user_id 持久化：短期对话（LangChain 消息）+ 长期简历解析档案。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from resume_agent.schemas import ParsedResume

LONG_TERM_FILENAME = "long_term.json"
SHORT_TERM_FILENAME = "short_term.json"
SHORT_TERM_MAX_MESSAGES = 80


def default_data_dir() -> Path:
    raw = os.environ.get("RESUME_AGENT_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".resume_agent"


def sanitize_user_id(user_id: str) -> str:
    s = user_id.strip()
    if not s:
        return "default"
    if s in {".", ".."}:
        raise ValueError("user_id 不能使用 . 或 ..")
    if not re.fullmatch(r"[\w.\-]{1,64}", s):
        raise ValueError("user_id 仅允许字母数字、下划线、点、连字符，长度 1～64")
    return s


def user_storage_dir(user_id: str) -> Path:
    return default_data_dir() / sanitize_user_id(user_id)


@dataclass(frozen=True)
class LongTermRecord:
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
    body = json.dumps(record.parsed_resume, ensure_ascii=False, indent=2)
    return (
        "【用户长期记忆·简历结构化档案（最近一次上传并解析）】\n"
        f"{body}\n"
        "说明：若用户口述与档案冲突，以当轮为准并提醒重新上传简历。\n"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
    _atomic_write_json(user_storage_dir(user_id) / LONG_TERM_FILENAME, record.to_json_dict())


def load_short_term_messages(user_id: str) -> list[BaseMessage]:
    path = user_storage_dir(user_id) / SHORT_TERM_FILENAME
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        raw = data["messages"]
    elif isinstance(data, list):
        raw = data
    else:
        return []
    try:
        return messages_from_dict(raw)
    except Exception:  # noqa: BLE001
        return []


def save_short_term_messages(user_id: str, messages: list[BaseMessage]) -> None:
    path = user_storage_dir(user_id) / SHORT_TERM_FILENAME
    trimmed = messages[-SHORT_TERM_MAX_MESSAGES:]
    serial = messages_to_dict(trimmed)
    _atomic_write_json(path, {"version": 1, "messages": serial})
