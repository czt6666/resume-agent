"""从消息列表中取最后一条 AI 文本。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage


def format_prior_turns_for_supervisor(messages: list[BaseMessage]) -> str:
    """把历史 Human/AI 轮次拼成主管可读的前序对话块（不含当前轮）。"""
    lines: list[str] = []
    for m in messages:
        role = getattr(m, "type", None) or ""
        if role == "human":
            lines.append(f"用户（前序）：{m.content}")
        elif role == "ai":
            lines.append(f"助手（前序）：{m.content}")
    if not lines:
        return ""
    return "\n\n".join(lines) + "\n\n"


def last_ai_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts: list[str] = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                return "\n".join(parts) if parts else str(c)
            return str(c)
    return ""
