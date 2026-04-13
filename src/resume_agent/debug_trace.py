"""调试输出：步骤轨迹 + 每次 Chat 模型调用的提示词（stderr）。"""

from __future__ import annotations

import os
import sys
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

_BAR = "=" * 72


def is_debug() -> bool:
    v = os.environ.get("RESUME_AGENT_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def trace_step(title: str, detail: str | None = None) -> None:
    """流程里程碑（不含完整 LLM 正文时可用短说明）。"""
    if not is_debug():
        return
    print(f"\n{_BAR}\n[resume-agent] ▶ {title}\n{_BAR}", file=sys.stderr)
    if detail:
        print(detail, file=sys.stderr)


def _msg_line(m: BaseMessage) -> str:
    role = type(m).__name__
    c = m.content
    if isinstance(c, str):
        body = c
    else:
        body = repr(c)
    return f"[{role}]\n{body}"


def _dump_chat_messages(tag: str, message_batches: list[list[BaseMessage]]) -> None:
    if not is_debug():
        return
    print(f"\n{_BAR}\n[resume-agent] LLM 提示词 · {tag}\n{_BAR}", file=sys.stderr)
    for bi, batch in enumerate(message_batches):
        print(f"--- batch {bi} ---", file=sys.stderr)
        for m in batch:
            print(_msg_line(m), file=sys.stderr)
            print(file=sys.stderr)


class LlmDebugCallbackHandler(BaseCallbackHandler):
    """在每次 Chat 模型调用前后打印入参消息与出参摘要。"""

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        name = (
            (serialized or {}).get("name")
            or (serialized or {}).get("id")
            or (kwargs.get("metadata") or {}).get("ls_model_name")
            or "chat_model"
        )
        _dump_chat_messages(str(name), messages)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if not is_debug():
            return
        texts: list[str] = []
        for gen_list in response.generations or []:
            for g in gen_list:
                t = getattr(g, "text", None) or ""
                if not t and getattr(g, "message", None) is not None:
                    c = g.message.content
                    t = c if isinstance(c, str) else str(c)
                t = (t or "").strip()
                if t:
                    texts.append(t)
        joined = "\n".join(texts)
        preview = joined[:1200] + ("…\n(输出已截断)" if len(joined) > 1200 else "")
        print(f"\n[resume-agent] LLM 回复摘要 ({len(joined)} 字)\n{preview}\n", file=sys.stderr)
