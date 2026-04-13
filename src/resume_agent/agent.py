"""对外 Agent 门面：多 Agent 编排（主管路由 + 子专员并行 + 汇总）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from resume_agent.message_utils import format_prior_turns_for_supervisor, last_ai_text
from resume_agent.orchestrator import run_resume_pipeline


def run_resume_orchestration(user_text: str) -> str:
    """执行完整简历优化流水线（主管 → 并行子 Agent → 合成）。"""
    return run_resume_pipeline(user_text)


def build_resume_agent():
    """保留兼容名：旧代码曾返回单体 create_agent 图。现已改为编排流水线，此函数返回 None。"""
    return None


def _last_ai_text(messages: list[BaseMessage]) -> str:
    return last_ai_text(messages)


def run_agent_turn(
    user_text: str,
    *,
    thread_messages: list[BaseMessage] | None = None,
) -> tuple[str, list[BaseMessage]]:
    """执行一轮对话，返回 (助手纯文本, 更新后的消息列表)。"""
    prior = format_prior_turns_for_supervisor(list(thread_messages or []))
    payload = f"{prior}【当前轮用户输入】\n{user_text}" if prior else user_text
    reply = run_resume_orchestration(payload)
    base: list[BaseMessage] = list(thread_messages or [])
    base.append(HumanMessage(content=user_text))
    base.append(AIMessage(content=reply))
    return reply, base
