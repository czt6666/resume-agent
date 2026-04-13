"""按 user_id 持久化：短期对话（LangChain 消息）+ 长期简历解析档案。

数据根目录由环境变量 RESUME_AGENT_DATA_DIR 指定；未设置时使用 ~/.resume_agent。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from resume_agent.schemas import ParsedResume

# 环境变量名（值应为目录路径）
RESUME_AGENT_DATA_DIR = "RESUME_AGENT_DATA_DIR"
LONG_TERM_FILENAME = "long_term.json"
SHORT_TERM_FILENAME = "short_term.json"
SHORT_TERM_MAX_MESSAGES = 80


def default_data_dir() -> Path:
    return Path(
        os.environ.get(RESUME_AGENT_DATA_DIR))
    ).expanduser().resolve()


def sanitize_user_id(user_id: str) -> str:
    if not isinstance(user_id, str):
        raise TypeError("user_id 必须是 str")
    return user_id.strip()


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
            version=int(data["version"]),
            updated_at=str(data["updated_at"]),
            resume_fingerprint=str(data["resume_fingerprint"]),
            parsed_resume=dict(data["parsed_resume"]),
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
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LongTermRecord.from_json_dict(data)
    except Exception:
        return None


def save_long_term(user_id: str, record: LongTermRecord) -> None:
    _atomic_write_json(user_storage_dir(user_id) / LONG_TERM_FILENAME, record.to_json_dict())


def load_short_term_messages(user_id: str) -> list[BaseMessage]:
    path = user_storage_dir(user_id) / SHORT_TERM_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return messages_from_dict(data["messages"])
    except Exception:
        return []


def save_short_term_messages(user_id: str, messages: list[BaseMessage]) -> None:
    path = user_storage_dir(user_id) / SHORT_TERM_FILENAME
    trimmed = messages[-SHORT_TERM_MAX_MESSAGES:]
    serial = messages_to_dict(trimmed)
    _atomic_write_json(path, {"version": 1, "messages": serial})
