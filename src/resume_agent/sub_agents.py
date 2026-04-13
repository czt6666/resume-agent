"""子 Agent：各方向专职的 create_agent 图（由主管并行调度）。"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from resume_agent.config import get_llm
from resume_agent.tools import (
    evaluate_github_project_candidates,
    search_github_fuzzy_for_resume,
    search_github_repositories,
    web_search,
)

_SUB_GRAPHS: dict[str, Any] = {}

PROJECT_SYSTEM = """你是「项目补强」专员，用简体中文回复。

职责：帮用户补齐或强化简历上的项目经历。可调用：
- `search_github_fuzzy_for_resume`：模糊方向时多路召回 GitHub；
- `search_github_repositories`：精确栈与场景时搜索（可用 language、min_stars、max_stars）；
- `evaluate_github_project_candidates`：必须对候选仓库做适配度评估；
- `web_search`：B 站/文章等（可加 site:bilibili.com）。

流程建议：召回 → 评估 → 必要时按评估建议继续搜索，直到有可行项目或说明原因。输出需包含：项目一句话、上手周期、技术点、简历帮助、含金量 1–10。禁止编造用户未做过的项目为已交付成果。
GitHub 频控时可提示配置环境变量 GITHUB_TOKEN。"""

TECH_SYSTEM = """你是「技术栈学习」专员，用简体中文回复。

职责：针对目标岗位补齐技术栈。主要用 `web_search` 找 B 站课程、优质文章、官方文档（可用 site: 限定）。

输出需说明：每条资源链接或标题、建议学到「了解 / 会用 / 深入原理」哪一档、对应能产出的作品或证明方式。不要替用户捏造已学会的技能。"""

EXPERIENCE_SYSTEM = """你是「实习/工作经历」专员，用简体中文回复。

职责：用 `web_search` 检索 BOSS 直聘等公开 JD，帮用户把经历写成 STAR、对齐关键词。**禁止**编造不存在的公司与任职事实。若涉及已倒闭公司，只指导如何如实表述与背调注意点。"""

COMPETITIVE_SYSTEM = """你是「竞争力评估」专员，用简体中文回复。

职责：根据上下文中的简历材料做多维度打分（结构、量化、技术匹配、项目深度等），并给应届生/同方向的**定性**对比；明确说明这是启发式评估。无需外网时可仅凭文本分析；若需市场侧参考可谨慎使用 `web_search`。"""

_KIND_CONFIG: dict[str, tuple[str, list]] = {
    "project": (
        PROJECT_SYSTEM,
        [
            search_github_fuzzy_for_resume,
            search_github_repositories,
            evaluate_github_project_candidates,
            web_search,
        ],
    ),
    "tech": (TECH_SYSTEM, [web_search]),
    "experience": (EXPERIENCE_SYSTEM, [web_search]),
    "competitiveness": (COMPETITIVE_SYSTEM, [web_search]),
}


def build_sub_agent_graph(kind: str) -> Any:
    """返回指定子专员的 LangGraph（create_agent），进程内缓存。"""
    if kind not in _KIND_CONFIG:
        raise ValueError(f"未知子专员: {kind}")
    if kind not in _SUB_GRAPHS:
        system, tools = _KIND_CONFIG[kind]
        _SUB_GRAPHS[kind] = create_agent(
            model=get_llm(),
            tools=tools,
            system_prompt=system,
            name=f"resume_sub_{kind}",
        )
    return _SUB_GRAPHS[kind]
