"""LangChain 1.x：使用 create_agent 构建简历优化 Agent。"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from resume_agent.config import get_llm
from resume_agent.tools import search_github_repositories, web_search

SYSTEM_PROMPT = """你是「简历优化」助手，用简体中文回复。

## 能力
1. **缺项目**：调用 `search_github_repositories` 找 GitHub 仓库；需要 B 站/文章时调用 `web_search`（可加 site:bilibili.com）。对每条推荐写清：项目一句话介绍、大致上手周期、涉及技术、对简历的帮助、含金量评分（1–10）及理由。可提「经典小项目」类型（如 CLI 工具、博客 API、爬虫管道）作为补充，但仓库以工具检索为准。
2. **缺技术栈**：用 `web_search` 找 B 站课程、博文或官方文档链接；说明建议学到什么程度：了解 / 会用 / 深入原理，并给出对应的学习产出（能做什么）。
3. **缺实习/工作经历**：用 `web_search` 检索 BOSS 直聘等**公开**岗位描述，帮用户对照 JD 写经历要点（STAR）；**禁止**编造不存在的公司与在职经历。若用户曾在真实存在但已注销/难查证的公司工作，只可指导如何**如实**撰写与背调注意事项，不得教唆造假。
4. **竞争力**：根据用户给出的简历文本做维度打分（结构、量化成果、技术匹配、项目深度等），并给出与常见应届生/同方向候选的**定性**对比；说明这是启发式评估而非精确统计。

## 原则
- 若用户消息中含「经 LLM 解析的简历结构化摘要」，请优先以其为索引定位经历/技能/短板，并与「简历原文」核对细节；摘要与原文冲突时以原文为准。
- 工具结果若为空，换关键词再搜或基于常识给保守建议。
- 回答分点清晰，可操作。"""


_agent_graph = None


def build_resume_agent():
    """返回 LangGraph 编译后的 Agent（LangChain 1.x create_agent）。进程内单例，避免重复编译图。"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent(
            model=get_llm(),
            tools=[search_github_repositories, web_search],
            system_prompt=SYSTEM_PROMPT,
            name="resume_optimizer",
        )
    return _agent_graph


def _last_ai_text(messages: list[BaseMessage]) -> str:
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


def run_agent_turn(
    user_text: str,
    *,
    thread_messages: list[BaseMessage] | None = None,
) -> tuple[str, list[BaseMessage]]:
    """执行一轮对话，返回 (助手纯文本, 更新后的消息列表)。"""
    graph = build_resume_agent()
    base: list[BaseMessage] = list(thread_messages or [])
    base.append(HumanMessage(content=user_text))
    result: dict[str, Any] = graph.invoke({"messages": base})
    out_messages: list[BaseMessage] = list(result.get("messages", base))
    return _last_ai_text(out_messages), out_messages
