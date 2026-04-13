"""简历优化：主管路由 → 子 create_agent 并行 → 汇总。单文件集中编排逻辑。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from resume_agent.config import get_llm
from resume_agent.tools import (
    evaluate_github_project_candidates,
    search_github_fuzzy_for_resume,
    search_github_repositories,
    web_search,
)

# —— 主管结构化输出 —— #
SubAgentKind = Literal["project", "tech", "experience", "competitiveness"]


class ResumeOrchestrationPlan(BaseModel):
    agents: list[SubAgentKind] = Field(default_factory=list, description="并行调用的专员 id，1～4 个")
    direct_response: str | None = Field(None, description="无需子专员时的直接答复；此时 agents 应为空")
    rationale: str = Field(default="", description="向用户说明分工理由，1～3 句")
    hints: dict[str, str] = Field(default_factory=dict, description="可选，键为专员 id")

    @model_validator(mode="after")
    def _normalize(self) -> ResumeOrchestrationPlan:
        seen: set[str] = set()
        uniq: list[SubAgentKind] = []
        for a in self.agents:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        self.agents = uniq
        dr = (self.direct_response or "").strip()
        self.direct_response = dr or None
        if self.direct_response and self.agents:
            self.direct_response = None
        if not self.agents and not self.direct_response:
            self.agents = ["project", "tech", "experience", "competitiveness"]
            extra = "（主管未指定路由，已默认并行启用四类专员。）"
            self.rationale = f"{self.rationale}\n{extra}".strip() if self.rationale else extra
        return self


# —— 消息小工具（CLI / 多轮） —— #
def format_prior_turns_for_supervisor(messages: list[BaseMessage]) -> str:
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


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = [
                    str(b.get("text", ""))
                    for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                return "\n".join(parts) if parts else str(c)
            return str(c)
    return ""


# —— 子 Agent —— #
_SUB: dict[str, Any] = {}

_SP_PROJECT = """你是「项目补强」专员，简体中文。用 search_github_fuzzy_for_resume / search_github_repositories /
evaluate_github_project_candidates / web_search。流程：召回→评估→必要时再搜。输出含项目一句话、周期、技术点、简历帮助、含金量1–10。勿编造用户未做过的交付。GitHub 频控可提示 GITHUB_TOKEN。"""

_SP_TECH = """你是「技术栈学习」专员，简体中文。主要 web_search（可 site:bilibili.com 等）。说明资源、学到了解/会用/深入原理哪一档、可产出什么。勿捏造已学会技能。"""

_SP_EXP = """你是「实习/工作经历」专员，简体中文。web_search 查公开 JD，STAR 对齐关键词。禁止编造公司与任职。已倒闭公司只谈如实表述与背调。"""

_SP_COMP = """你是「竞争力评估」专员，简体中文。据简历材料多维度打分与定性对比（启发式）。可仅用文本或谨慎 web_search。"""

_SUB_CFG: dict[str, tuple[str, list]] = {
    "project": (
        _SP_PROJECT,
        [
            search_github_fuzzy_for_resume,
            search_github_repositories,
            evaluate_github_project_candidates,
            web_search,
        ],
    ),
    "tech": (_SP_TECH, [web_search]),
    "experience": (_SP_EXP, [web_search]),
    "competitiveness": (_SP_COMP, [web_search]),
}

_LABELS = {
    "project": "项目补强",
    "tech": "技术栈与学习路径",
    "experience": "实习/工作经历",
    "competitiveness": "竞争力评估",
}


def _sub_graph(kind: str) -> Any:
    if kind not in _SUB_CFG:
        raise ValueError(f"未知专员: {kind}")
    if kind not in _SUB:
        sys_p, tools = _SUB_CFG[kind]
        _SUB[kind] = create_agent(
            model=get_llm(),
            tools=tools,
            system_prompt=sys_p,
            name=f"resume_sub_{kind}",
        )
    return _SUB[kind]


_SUP = """你是简历优化「主管」，只做调度。阅读完整用户上下文，决定并行启用哪些专员：
project=缺项目/GitHub练手；tech=缺技术栈与学习资源；experience=弱经历/JD对齐；competitiveness=打分与对比。
问题窄则少激活；可设 agents 为空并填 direct_response 直接答。rationale 1～3 句；hints 可选。"""

_SYN = """你是汇总专员。把主管说明与子专员输出合并为一份中文回复：去重、二级标题、保留链接，语气专业。"""


def _plan(payload: str) -> ResumeOrchestrationPlan:
    llm = get_llm().with_structured_output(ResumeOrchestrationPlan)
    out = llm.invoke(
        [HumanMessage(content=f"{_SUP}\n\n--- 上下文 ---\n{payload}\n--- 结束 ---")],
    )
    if not isinstance(out, ResumeOrchestrationPlan):
        raise TypeError(type(out))
    return out


def _run_sub(kind: str, payload: str, plan: ResumeOrchestrationPlan) -> tuple[str, str]:
    g = _sub_graph(kind)
    hint = (plan.hints or {}).get(kind, "").strip()
    task = f"""{payload}

---
【调度】专员={kind}（{_LABELS[kind]}） 主管：{plan.rationale or "无"} 焦点：{hint or "请据上下文自行判断"}
请只完成本专员范围，Markdown 分点。"""
    r = g.invoke({"messages": [HumanMessage(content=task)]})
    return kind, _last_ai_text(list(r.get("messages", [])))


def _parallel(plan: ResumeOrchestrationPlan, payload: str) -> dict[str, str]:
    kinds = list(plan.agents)
    if not kinds:
        return {}
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(kinds))) as ex:
        futs = {ex.submit(_run_sub, k, payload, plan): k for k in kinds}
        for fut in as_completed(futs):
            k0 = futs[fut]
            try:
                k, text = fut.result()
                out[k] = text
            except Exception as e:  # noqa: BLE001
                out[k0] = f"（{k0} 出错：{e}）"
    return out


def _merge(plan: ResumeOrchestrationPlan, sub: dict[str, str]) -> str:
    order = ("project", "tech", "experience", "competitiveness")
    parts = [f"### {_LABELS[k]}\n{v.strip()}" for k in order if (v := sub.get(k, "")) and v.strip()]
    body = "\n\n".join(parts) if parts else "（子专员无有效输出。）"
    resp = get_llm().invoke(
        [
            SystemMessage(content=_SYN),
            HumanMessage(
                content=f"【主管】{plan.rationale or '无'}\n\n【子输出】\n{body}\n\n请输出最终回复。",
            ),
        ],
    )
    c = resp.content
    return c if isinstance(c, str) else str(c)


def run_resume_pipeline(full_user_payload: str) -> str:
    plan = _plan(full_user_payload)
    if plan.direct_response:
        h = f"【主管】{plan.rationale}\n\n" if plan.rationale else ""
        return f"{h}{plan.direct_response.strip()}"
    return _merge(plan, _parallel(plan, full_user_payload))


def run_resume_pipeline_debug(full_user_payload: str) -> dict[str, Any]:
    plan = _plan(full_user_payload)
    if plan.direct_response:
        return {
            "plan": plan.model_dump(),
            "direct": True,
            "final": (f"【主管】{plan.rationale}\n\n" if plan.rationale else "")
            + plan.direct_response.strip(),
        }
    s = _parallel(plan, full_user_payload)
    return {"plan": plan.model_dump(), "direct": False, "sub_results": s, "final": _merge(plan, s)}


def run_resume_orchestration(user_text: str) -> str:
    return run_resume_pipeline(user_text)


def build_resume_agent():
    """兼容旧 API；编排模式下无单体图。"""
    return None


def run_agent_turn(
    user_text: str,
    *,
    thread_messages: list[BaseMessage] | None = None,
) -> tuple[str, list[BaseMessage]]:
    prior = format_prior_turns_for_supervisor(list(thread_messages or []))
    payload = f"{prior}【当前轮用户输入】\n{user_text}" if prior else user_text
    reply = run_resume_pipeline(payload)
    base = list(thread_messages or [])
    base.extend([HumanMessage(content=user_text), AIMessage(content=reply)])
    return reply, base
