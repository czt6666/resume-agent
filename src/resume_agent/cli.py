"""命令行入口：uv run resume-agent 或 python -m resume_agent.cli"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from resume_agent.agent import format_prior_turns_for_supervisor, run_resume_orchestration
from resume_agent.loaders import ResumeLoadError, load_resume_text
from resume_agent.memory import (
    LongTermRecord,
    load_long_term,
    load_short_term_messages,
    long_term_prompt_block,
    save_long_term,
    save_short_term_messages,
    sanitize_user_id,
)
from resume_agent.parser import build_agent_context, parse_resume_with_llm

DEFAULT_USER_QUESTION = "请帮我分析一下这份简历"


@dataclass(frozen=True)
class CliConfig:
    """归一化后的命令行参数（main 只依赖本结构，不读 argparse Namespace）。"""

    user_id: str
    resume_path: Path | None
    user_question: str


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="简历优化 Agent（LangChain 1.x create_agent）")
    p.add_argument(
        "-r",
        "--resume",
        type=Path,
        default=None,
        metavar="FILE",
        dest="resume_path",
        help="简历文件路径：上传后先用 LLM 解析，再进入优化对话（.txt / .md / .pdf）",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="调试：步骤与每次 LLM 提示词/回复摘要输出到 stderr（等同环境变量 RESUME_AGENT_DEBUG=1）",
    )
    p.add_argument(
        "--user-id",
        default="default",
        metavar="ID",
        dest="user_id",
        help="用户标识，用于短期对话记忆与长期简历档案目录（默认 default，可用环境变量 RESUME_AGENT_DATA_DIR 指定根目录）",
    )
    p.add_argument(
        "message",
        nargs="?",
        default=None,
        help='用户问题；省略时默认为「请帮我分析一下这份简历」。',
    )
    return p


def parse_cli(argv: list[str] | None = None) -> CliConfig:
    """解析 argv 并归一化为 CliConfig。"""
    ns = _build_arg_parser().parse_args(argv)
    if ns.debug:
        os.environ["RESUME_AGENT_DEBUG"] = "1"
    return CliConfig(
        user_id=sanitize_user_id(ns.user_id),
        resume_path=ns.resume_path,
        user_question=(ns.message or "").strip() or DEFAULT_USER_QUESTION,
    )


def _invoke_agent_with_memory(user_id: str, user_message_content: str) -> str:
    """短期记忆：拼接历史为单条上下文后走主管编排流水线，并写回对话列表。"""
    prior = load_short_term_messages(user_id)
    history_block = format_prior_turns_for_supervisor(prior)
    payload = f"{history_block}【当前轮用户输入】\n{user_message_content}"
    reply = run_resume_orchestration(payload)
    out_messages = list(prior)
    out_messages.append(HumanMessage(content=user_message_content))
    out_messages.append(AIMessage(content=reply))
    save_short_term_messages(user_id, out_messages)
    return reply


def _run_resume_and_agent(user_id: str, raw: str, parsed, user_question: str) -> None:
    save_long_term(user_id, LongTermRecord.from_parsed(raw, parsed))
    ctx = build_agent_context(raw, parsed)
    payload = f"{ctx}\n\n【用户问题】\n{user_question}"
    print(_invoke_agent_with_memory(user_id, payload))


def _run_plain_agent(user_id: str, user_question: str) -> None:
    lt = load_long_term(user_id)
    prefix = long_term_prompt_block(lt) if lt else ""
    payload = f"{prefix}【用户问题】\n{user_question}"
    print(_invoke_agent_with_memory(user_id, payload))


def main() -> None:
    cfg = parse_cli()
    if cfg.resume_path is not None:
        try:
            raw = load_resume_text(cfg.resume_path)
        except ResumeLoadError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        try:
            parsed = parse_resume_with_llm(raw)
        except Exception as e:  # noqa: BLE001
            print(f"LLM 解析失败: {e}", file=sys.stderr)
            sys.exit(3)
        _run_resume_and_agent(cfg.user_id, raw, parsed, cfg.user_question)
        return
    _run_plain_agent(cfg.user_id, cfg.user_question)


if __name__ == "__main__":
    main()
