"""命令行入口：uv run resume-agent 或 python -m resume_agent.cli"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from resume_agent.agent import build_resume_agent, _last_ai_text
from resume_agent.loaders import ResumeLoadError, load_resume_text
from resume_agent.parser import build_agent_context, parse_resume_with_llm, parsed_resume_to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="简历优化 Agent（LangChain 1.x create_agent）")
    parser.add_argument(
        "-r",
        "--resume",
        type=Path,
        default=None,
        metavar="FILE",
        help="简历文件路径：上传后先用 LLM 解析，再进入优化对话（.txt / .md / .pdf）",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="仅调用 LLM 解析简历并输出 JSON，不运行优化 Agent",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="用户问题；与 --resume 连用时写在解析之后。省略且未指定 --resume 时从 stdin 读入",
    )
    args = parser.parse_args()

    message = args.message
    if message is None and args.resume is None and not sys.stdin.isatty():
        message = sys.stdin.read().strip() or None

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
            print(parsed_resume_to_json(parsed))
            return
        ctx = build_agent_context(raw, parsed)
        user_q = message or "请全面分析这份简历的优缺点，并给出可执行的优化建议（含缺项补强思路）。"
        text = f"{ctx}\n\n【用户问题】\n{user_q}"
    else:
        if args.parse_only:
            print("需要同时指定 --resume 文件才能使用 --parse-only", file=sys.stderr)
            sys.exit(2)
        text = message
        if text is None:
            text = sys.stdin.read().strip()
        if not text:
            parser.print_help()
            sys.exit(1)

    graph = build_resume_agent()
    result = graph.invoke({"messages": [HumanMessage(content=text)]})
    print(_last_ai_text(list(result.get("messages", []))))


if __name__ == "__main__":
    main()
