"""按 user_id 持久化：短期对话历史（LangChain 消息）+ 长期简历档案（见 user_profile_store）。"""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from resume_agent.user_profile_store import (
    LONG_TERM_FILENAME,
    LongTermRecord,
    default_data_dir,
    load_long_term,
    long_term_prompt_block,
    save_long_term,
    sanitize_user_id,
    user_storage_dir,
)

SHORT_TERM_FILENAME = "short_term.json"
SHORT_TERM_MAX_MESSAGES = 80

__all__ = [
    "LONG_TERM_FILENAME",
    "SHORT_TERM_FILENAME",
    "SHORT_TERM_MAX_MESSAGES",
    "LongTermRecord",
    "default_data_dir",
    "load_long_term",
    "load_short_term_messages",
    "long_term_prompt_block",
    "save_long_term",
    "save_short_term_messages",
    "sanitize_user_id",
    "user_storage_dir",
]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "messages": serial}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
