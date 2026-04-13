"""主管 Agent 路由 + 子 Agent 并行执行 + 汇总合成。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage

from resume_agent.config import get_llm
from resume_agent.message_utils import last_ai_text
from resume_agent.schemas import ResumeOrchestrationPlan
from resume_agent.sub_agents import build_sub_agent_graph

_SUPERVISOR_INSTRUCTION = """你是简历优化系统的「主管」，只做调度决策，不自己写长篇优化方案。

阅读下方【完整用户上下文】（可含简历解析 JSON、原文、用户问题、长期记忆等），判断应并行启用哪些子专员：

- project：缺项目、项目太水、需要 GitHub 练手方向或项目评估。
- tech：缺技术栈、要学新栈、需要教程/文档/课程线索。
- experience：缺或弱实习/工作经历、需要对照 JD、STAR 改写（不得造假）。
- competitiveness：需要简历打分、与同龄人/同方向候选的定性对比。

规则：
1. 用户问题很窄时只激活相关专员（可 1 个）；需要全面体检时可激活多个。
2. 若只需简短答复、或上下文严重不足无法分派，设 agents 为空，并填写 direct_response 直接给用户。
3. rationale 用 1～3 句中文向用户说明分工理由。
4. hints 可选：给某个专员的焦点句（键为 project / tech / experience / competitiveness）。

输出必须符合结构化模式（agents、direct_response、rationale、hints）。"""

_SYNTH_SYSTEM = """你是简历优化系统的「汇总专员」。将主管分工说明与多位子专员的输出合并为**一份**给用户的中文回复。

要求：去重、结构清晰（可用二级标题）、保留可执行建议与链接；不要重复粘贴同一链接；语气专业友好。若某子专员输出为空或失败，可略过并注明。"""

_AGENT_LABELS = {
    "project": "项目补强",
    "tech": "技术栈与学习路径",
    "experience": "实习/工作经历",
    "competitiveness": "竞争力评估",
}


def plan_tasks(full_user_payload: str) -> ResumeOrchestrationPlan:
    llm = get_llm().with_structured_output(ResumeOrchestrationPlan)
    msg = HumanMessage(
        content=f"{_SUPERVISOR_INSTRUCTION}\n\n--- 完整用户上下文 ---\n{full_user_payload}\n--- 结束 ---",
    )
    plan = llm.invoke([msg])
    if not isinstance(plan, ResumeOrchestrationPlan):
        raise TypeError(f"主管结构化输出异常: {type(plan)}")
    return plan


def _run_one_sub_agent(kind: str, full_payload: str, plan: ResumeOrchestrationPlan) -> tuple[str, str]:
    graph = build_sub_agent_graph(kind)
    hint = (plan.hints or {}).get(kind, "").strip()
    task = f"""以下是用户与简历相关的完整上下文（与主管所见一致）：

{full_payload}

---
【主管调度】
- 本轮你的专员类型：{kind}（{_AGENT_LABELS.get(kind, kind)}）
- 主管说明：{plan.rationale or "无"}
- 给你的焦点提示：{hint or "无；请根据上下文自行判断重点。"}

请只完成本专员职责范围内的建议，使用 Markdown 分点，勿代其他专员作答。"""
    result = graph.invoke({"messages": [HumanMessage(content=task)]})
    msgs = list(result.get("messages", []))
    return kind, last_ai_text(msgs)


def run_sub_agents_parallel(plan: ResumeOrchestrationPlan, full_payload: str) -> dict[str, str]:
    kinds = list(plan.agents)
    if not kinds:
        return {}
    max_workers = min(8, max(1, len(kinds)))
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_sub_agent, k, full_payload, plan): k for k in kinds}
        for fut in as_completed(futures):
            kind = futures[fut]
            try:
                k2, text = fut.result()
                out[k2] = text
            except Exception as e:  # noqa: BLE001
                out[kind] = f"（{kind} 专员执行出错：{e}）"
    return out


def synthesize(plan: ResumeOrchestrationPlan, sub_results: dict[str, str]) -> str:
    order = ["project", "tech", "experience", "competitiveness"]
    sections: list[str] = []
    for k in order:
        if k in sub_results and sub_results[k].strip():
            sections.append(f"### {_AGENT_LABELS[k]}\n{sub_results[k].strip()}")
    body = "\n\n".join(sections) if sections else "（子专员未返回有效内容。）"

    llm = get_llm()
    human = HumanMessage(
        content=(
            f"【主管说明】\n{plan.rationale or '无'}\n\n"
            f"【子专员原始输出】\n{body}\n\n请输出合并后的最终回复。"
        ),
    )
    resp = llm.invoke([SystemMessage(content=_SYNTH_SYSTEM), human])
    c = resp.content
    return c if isinstance(c, str) else str(c)


def run_resume_pipeline(full_user_payload: str) -> str:
    """对外主入口：主管决策 → 并行子 Agent → 汇总。"""
    plan = plan_tasks(full_user_payload)
    if plan.direct_response:
        header = f"【主管】{plan.rationale}\n\n" if plan.rationale else ""
        return f"{header}{plan.direct_response.strip()}"

    sub = run_sub_agents_parallel(plan, full_user_payload)
    return synthesize(plan, sub)


def run_resume_pipeline_debug(full_user_payload: str) -> dict[str, object]:
    """调试：返回计划、各子专员原文与最终稿。"""
    plan = plan_tasks(full_user_payload)
    if plan.direct_response:
        return {
            "plan": plan.model_dump(),
            "direct": True,
            "final": (f"【主管】{plan.rationale}\n\n" if plan.rationale else "") + plan.direct_response.strip(),
        }
    sub = run_sub_agents_parallel(plan, full_user_payload)
    final = synthesize(plan, sub)
    return {
        "plan": plan.model_dump(),
        "direct": False,
        "sub_results": sub,
        "final": final,
    }
