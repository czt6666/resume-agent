"""使用 LLM 将简历原文解析为结构化数据（LangChain 1.x + structured output）。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from resume_agent.config import get_llm
from resume_agent.debug_trace import trace_step
from resume_agent.schemas import ParsedResume

_PARSE_INSTRUCTION = """你是简历解析器。根据用户给出的「简历全文」，抽取结构化信息，用中文填写各字段。
规则：
- 未出现的信息填 null 或空列表，不要编造经历。
- contact_masked 只说明是否包含邮箱/电话等类型，用一句话概括，不要输出真实邮箱或完整手机号。
- gaps_or_notes 写你从文本中能看出的短板（缺实习、项目描述过短、无数据指标等）。
- skills 用简短词或短语列表（如 Python、FastAPI）。
- 工作/项目 bullet 尽量保留原文要点，可适当合并重复表述。"""


def parse_resume_with_llm(resume_plain_text: str) -> ParsedResume:
    """调用 LLM 将简历纯文本解析为 `ParsedResume`。"""
    text = resume_plain_text.strip()
    if not text:
        raise ValueError("简历正文为空")

    trace_step("简历解析：LLM 结构化抽取 ParsedResume")
    llm = get_llm()
    structured = llm.with_structured_output(ParsedResume)
    msg = HumanMessage(
        content=f"{_PARSE_INSTRUCTION}\n\n--- 简历全文 ---\n{text}\n--- 结束 ---"
    )
    result = structured.invoke([msg])
    if not isinstance(result, ParsedResume):
        raise TypeError(f"结构化输出类型异常: {type(result)}")
    return result


def parsed_resume_to_json(resume: ParsedResume, *, indent: int = 2) -> str:
    """便于打印或传入下游的 JSON 字符串。"""
    return resume.model_dump_json(ensure_ascii=False, indent=indent)


def parse_then_optimize(
    resume_plain_text: str,
    user_message: str | None = None,
) -> tuple[ParsedResume, str]:
    """先 LLM 解析简历，再调用优化 Agent 返回 (解析结果, 助手回复)。"""
    from resume_agent.agent import run_agent_turn

    parsed = parse_resume_with_llm(resume_plain_text)
    ctx = build_agent_context(resume_plain_text, parsed)
    q = user_message or "请帮我分析一下这份简历"
    reply, _ = run_agent_turn(f"{ctx}\n\n【用户问题】\n{q}")
    return parsed, reply


def build_agent_context(resume_plain_text: str, parsed: ParsedResume, max_raw_chars: int = 14_000) -> str:
    """把解析结果 + 截断后的原文拼成发给优化 Agent 的上下文。"""
    raw = resume_plain_text.strip()
    tail = "\n\n…（原文已截断）" if len(raw) > max_raw_chars else ""
    raw_excerpt = raw[:max_raw_chars] + tail

    structured = parsed.model_dump()
    return (
        "【经 LLM 解析的简历结构化摘要】\n"
        f"{json.dumps(structured, ensure_ascii=False, indent=2)}\n\n"
        "【简历原文】\n"
        f"{raw_excerpt}"
    )
