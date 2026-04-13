"""命令行入口：uv run resume-agent 或 python -m resume_agent.cli"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from resume_agent.agent import run_resume_orchestration
from resume_agent.message_utils import format_prior_turns_for_supervisor
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
from resume_agent.parser import build_agent_context, parse_resume_with_llm, parsed_resume_to_json


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="简历优化 Agent（LangChain 1.x create_agent）")
    p.add_argument(
        "-r",
        "--resume",
        type=Path,
        default=None,
        metavar="FILE",
        help="简历文件路径：上传后先用 LLM 解析，再进入优化对话（.txt / .md / .pdf）",
    )
    p.add_argument(
        "--user-id",
        default="default",
        metavar="ID",
        help="用户标识，用于短期对话记忆与长期简历档案目录（默认 default，可用环境变量 RESUME_AGENT_DATA_DIR 指定根目录）",
    )
    p.add_argument(
        "--parse-only",
        action="store_true",
        help="仅调用 LLM 解析简历并输出 JSON，不运行优化 Agent",
    )
    p.add_argument(
        "message",
        nargs="?",
        default=None,
        help="用户问题；与 --resume 连用时写在解析之后。省略且未指定 --resume 时从 stdin 读入",
    )
    return p


def _resolve_cli_message(args: argparse.Namespace) -> str | None:
    """确定单轮用户输入文本（不含简历上下文）。"""
    message = args.message
    if message is None and args.resume is None and not sys.stdin.isatty():
        message = sys.stdin.read().strip() or None
    return message


def _read_plain_user_text(message: str | None) -> str:
    text = message
    if text is None:
        text = sys.stdin.read().strip()
    if not text:
        raise ValueError("缺少用户问题：请提供参数、管道输入或在终端输入内容。")
    return text


def _invoke_agent_with_memory(user_id: str, user_message_content: str) -> str:
    """短期记忆：拼接历史为单条上下文后走主管编排流水线，并写回对话列表。"""
    prior = load_short_term_messages(user_id)
    history_block = format_prior_turns_for_supervisor(prior)
    payload = (
        f"{history_block}【当前轮用户输入】\n{user_message_content}"
        if history_block
        else user_message_content
    )
    reply = run_resume_orchestration(payload)
    out_messages = list(prior)
    out_messages.append(HumanMessage(content=user_message_content))
    out_messages.append(AIMessage(content=reply))
    save_short_term_messages(user_id, out_messages)
    return reply


def _persist_resume_long_term(user_id: str, raw: str, parsed) -> None:
    """长期记忆：每次成功解析简历后覆盖写入（含指纹，便于识别是否换过文件）。"""
    save_long_term(user_id, LongTermRecord.from_parsed(raw, parsed))


def _run_parse_only(user_id: str, raw: str, parsed) -> None:
    print(parsed_resume_to_json(parsed))
    _persist_resume_long_term(user_id, raw, parsed)


def _run_resume_and_agent(user_id: str, raw: str, parsed, user_question: str) -> None:
    _persist_resume_long_term(user_id, raw, parsed)
    ctx = build_agent_context(raw, parsed)
    payload = f"{ctx}\n\n【用户问题】\n{user_question}"
    print(_invoke_agent_with_memory(user_id, payload))


def _run_plain_agent(user_id: str, user_question: str) -> None:
    lt = load_long_term(user_id)
    prefix = long_term_prompt_block(lt) if lt else ""
    payload = f"{prefix}【用户问题】\n{user_question}" if prefix else user_question
    print(_invoke_agent_with_memory(user_id, payload))


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        user_id = sanitize_user_id(args.user_id)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    message = _resolve_cli_message(args)

    if args.resume is not None:
        try:
            raw = load_resume_text(args.resume)
        except ResumeLoadError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        try:
            parsed = parse_resume_with_llm(raw)
        except Exception as e:  # noqa: BLE001
            print(f"LLM 解析失败: {e}", file=sys.stderr)
            sys.exit(3)

        if args.parse_only:
            _run_parse_only(user_id, raw, parsed)
            return

        default_q = "请全面分析这份简历的优缺点，并给出可执行的优化建议（含缺项补强思路）。"
        user_q = message or default_q
        _run_resume_and_agent(user_id, raw, parsed, user_q)
        return

    if args.parse_only:
        print("需要同时指定 --resume 文件才能使用 --parse-only", file=sys.stderr)
        sys.exit(2)

    try:
        user_text = _read_plain_user_text(message)
    except ValueError:
        parser.print_help()
        sys.exit(1)

    _run_plain_agent(user_id, user_text)


if __name__ == "__main__":
    main()
