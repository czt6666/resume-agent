"""供 Agent 调用的外部检索工具（GitHub / 网页搜索）。"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage

from resume_agent.schemas import GithubFitEvaluation

GITHUB_API = "https://api.github.com"


def _bmp_text(s: str, max_len: int | None = None) -> str:
    """去掉基本多文种平面之外的字符（常见 emoji），避免部分 Windows 终端编码报错。"""
    t = "".join(c for c in s if ord(c) <= 0xFFFF)
    if max_len is not None:
        t = t[:max_len]
    return t.strip()


def _github_headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "resume-agent/0.1 (resume optimization helper)",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _github_search_items(
    query: str,
    *,
    sort: str = "stars",
    order: str = "desc",
    per_page: int = 10,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """返回 (items, error_message)。error 时 items 为 None。"""
    per_page = max(1, min(int(per_page), 10))
    params = {"q": query, "sort": sort, "order": order, "per_page": str(per_page)}
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(
                f"{GITHUB_API}/search/repositories",
                params=params,
                headers=_github_headers(),
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        return None, str(e)

    return list(data.get("items") or []), None


def _format_github_items(items: list[dict[str, Any]], start_index: int = 1) -> str:
    lines: list[str] = []
    for i, repo in enumerate(items, start_index):
        name = repo.get("full_name", "")
        desc = _bmp_text((repo.get("description") or "").replace("\n", " "), 500)
        stars = repo.get("stargazers_count", 0)
        lang = repo.get("language") or "—"
        url = repo.get("html_url", "")
        topics = ", ".join(repo.get("topics") or [])[:120]
        lines.append(
            f"{i}. {name} ({lang}, stars:{stars})\n   {desc}\n   {url}\n   topics: {topics or '—'}"
        )
    return "\n\n".join(lines)


def search_github_repositories(
    query: str,
    max_results: int = 5,
    language: str = "",
    min_stars: int | None = None,
    max_stars: int | None = None,
) -> str:
    """在 GitHub 上按关键词搜索公开仓库（按 star 排序），用于为用户推荐可写进简历的项目。

    参数:
        query: 核心搜索词，勿重复写入 language/stars（会用参数拼接），如 \"web scraper api\"。
        max_results: 返回条数，默认 5，最大 10。
        language: 可选，如 Python、Go、TypeScript（写入 GitHub 的 language: 限定）。
        min_stars / max_stars: 可选，须同时传入，写入 GitHub stars:lo..hi。

    返回:
        仓库列表的简要说明（名称、URL、简介、star 数、主要语言）。
    """
    max_results = max(1, min(int(max_results), 10))
    q = query.strip()
    if language.strip():
        q = f"{q} language:{language.strip()}"
    if min_stars is not None and max_stars is not None:
        q = f"{q} stars:{int(min_stars)}..{int(max_stars)}"
    q = f"{q} archived:false"

    items, err = _github_search_items(q, sort="stars", order="desc", per_page=max_results)
    if err:
        return f"GitHub 搜索失败: {err}"
    if not items:
        return "未找到匹配的仓库，可换一个更具体的关键词（技术栈 + 场景，如 fastapi blog）。"
    return _format_github_items(items)


def search_github_fuzzy_for_resume(
    keywords: str,
    language: str = "",
    min_stars: int = 50,
    max_stars: int = 12000,
    per_variant: int = 5,
) -> str:
    """同一查询串做两次检索（按 star / 按最近更新），结果去重合并。配合 evaluate_github_project_candidates 筛选。

    参数:
        keywords: 用户意图关键词（建议含技术栈+场景）。
        language: 可选，如 Python。
        min_stars / max_stars: star 区间。
        per_variant: 每次请求条数（最大 10）。

    返回:
        去重后的仓库列表，[来源] 为 stars 或 updated。
    """
    per_variant = max(1, min(int(per_variant), 10))
    lang = language.strip()
    lo, hi = int(min_stars), int(max_stars)

    parts = [keywords.strip(), f"stars:{lo}..{hi}", "archived:false"]
    if lang:
        parts.insert(1, f"language:{lang}")
    q = " ".join(parts)

    seen: set[str] = set()
    merged: list[tuple[dict[str, Any], str]] = []

    for i, sort in enumerate(("stars", "updated")):
        if i:
            time.sleep(0.25)
        items, err = _github_search_items(q, sort=sort, order="desc", per_page=per_variant)
        if err or not items:
            continue
        for repo in items:
            fn = repo.get("full_name") or ""
            if not fn or fn in seen:
                continue
            seen.add(fn)
            merged.append((repo, sort))
            if len(merged) >= 15:
                break
        if len(merged) >= 15:
            break

    if not merged:
        return "多路搜索无结果。可换英文关键词或调整 stars 区间。"

    lines: list[str] = []
    for i, (repo, label) in enumerate(merged, 1):
        name = repo.get("full_name", "")
        desc = _bmp_text((repo.get("description") or "").replace("\n", " "), 500)
        stars = repo.get("stargazers_count", 0)
        lang_o = repo.get("language") or "—"
        url = repo.get("html_url", "")
        topics = ", ".join(repo.get("topics") or [])[:120]
        lines.append(
            f"{i}. {name} ({lang_o}, stars:{stars}) [来源:{label}]\n   {desc}\n   {url}\n   topics: {topics or '—'}"
        )
    return "\n\n".join(lines)


def evaluate_github_project_candidates(user_context: str, github_candidates_text: str) -> str:
    """对一批 GitHub 仓库做「简历练手项目」适配度评估，并给出是否继续搜索的建议。

    根据用户背景（年级、已会栈、目标岗位、可投入时间）判断每个仓库是过易、合适还是过难，
    并给出 1～10 的难度与简历价值分。若整体不合适，必须在 next_search_queries 中给出新的搜索 query，
    供你调用 `search_github_repositories` 或 `search_github_fuzzy_for_resume` 继续检索。

    参数:
        user_context: 用户水平与目标简述（越具体越好）。
        github_candidates_text: 上游工具返回的仓库列表原文（含序号与 URL）。

    返回:
        结构化评估文本：逐条 verdict、理由、以及建议的下一轮 GitHub 搜索 query。
    """
    raw = (github_candidates_text or "").strip()
    if not raw:
        return (
            "没有候选仓库文本。请先调用 search_github_fuzzy_for_resume 或 "
            "search_github_repositories，再把工具输出交给本工具。"
        )

    from resume_agent.config import get_llm
    from resume_agent.debug_trace import trace_step

    trace_step("工具 evaluate_github_project_candidates：LLM 评估候选仓库")
    system = SystemMessage(
        content=(
            "你是资深工程师与校招/社招简历顾问。任务：判断 GitHub 仓库是否适合作为"
            "「能写进简历、难度适中」的练手项目。\n"
            "too_easy：主要是 awesome-list、纯 CRUD 脚手架、几行 demo、或无需理解即可照抄。\n"
            "ok：中等规模，能在 1～4 周内做出可讲故事的产出，技术点可写 bullet。\n"
            "too_hard：操作系统/编译器/大型分布式等需长期投入，或文档极差无从下手。\n"
            "unclear：信息不足时保守标记。\n"
            "若 ok 的仓库少于 1 个，或整体偏题，必须填写 next_search_queries（1～3 条），"
            "query 要具体（技术栈+场景），可含 language:、stars:数字..数字、-topic:awesome 等 GitHub 语法。"
        )
    )
    human = HumanMessage(
        content=f"【用户背景】\n{user_context.strip()}\n\n【候选仓库】\n{raw}",
    )
    llm = get_llm().with_structured_output(GithubFitEvaluation)
    try:
        ev: GithubFitEvaluation = llm.invoke([system, human])
    except Exception as e:  # noqa: BLE001
        return f"LLM 评估失败: {e}"

    parts: list[str] = [f"【小结】{ev.summary}", "", "【逐条】"]
    for row in ev.rows:
        parts.append(
            f"- {row.repo_ref} → {row.verdict} | 难度{row.difficulty_1_10}/10 "
            f"简历价值{row.resume_value_1_10}/10\n  {row.reason}"
        )
    if ev.next_search_queries:
        parts.extend(["", "【建议下一轮 GitHub 搜索 query】", *[f"- {nq}" for nq in ev.next_search_queries]])
    return "\n".join(parts)


def web_search(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 做网页摘要搜索。用于查找 B 站教程、博客、官方文档、BOSS 直聘上的公开职位描述等。

    参数:
        query: 完整搜索语句。可用 site: 限定站点，例如：
            - \"site:bilibili.com FastAPI 入门\"
            - \"site:zhipin.com Python 实习\"
            - \"FastAPI 官方文档\"
        max_results: 返回条数，默认 5。

    返回:
        搜索结果的标题与摘要文本（可能不完整，仅供写作参考）。
    """
    max_results = max(1, min(int(max_results), 10))
    rows: list[str] = []
    try:
        with DDGS() as ddgs:
            gen = ddgs.text(query, max_results=max_results) or []
            for j, item in enumerate(gen, 1):
                title = _bmp_text(item.get("title", ""), 200)
                body = _bmp_text((item.get("body") or "").replace("\n", " "), 400)
                href = item.get("href", "")
                rows.append(f"{j}. {title}\n   {body}\n   {href}")
    except Exception as e:  # noqa: BLE001
        return f"网页搜索失败: {e}"

    if not rows:
        return "搜索无结果。"
    return "\n\n".join(rows)
