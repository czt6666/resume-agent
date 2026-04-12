"""供 Agent 调用的外部检索工具（GitHub / 网页搜索）。"""

from __future__ import annotations

from typing import Any

import httpx

GITHUB_API = "https://api.github.com"


def _bmp_text(s: str, max_len: int | None = None) -> str:
    """去掉基本多文种平面之外的字符（常见 emoji），避免部分 Windows 终端编码报错。"""
    t = "".join(c for c in s if ord(c) <= 0xFFFF)
    if max_len is not None:
        t = t[:max_len]
    return t.strip()


def search_github_repositories(query: str, max_results: int = 5) -> str:
    """在 GitHub 上按关键词搜索公开仓库（按 star 排序），用于为用户推荐可写进简历的项目。

    参数:
        query: 搜索词，可含语言等，如 \"language:Python web scraper\"。
        max_results: 返回条数，默认 5，最大建议 10。

    返回:
        仓库列表的简要说明（名称、URL、简介、star 数、主要语言）。
    """
    max_results = max(1, min(int(max_results), 10))
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": str(max_results)}
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "resume-agent/0.1 (resume optimization helper)",
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(f"{GITHUB_API}/search/repositories", params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        return f"GitHub 搜索失败: {e}"

    items: list[dict[str, Any]] = data.get("items") or []
    if not items:
        return "未找到匹配的仓库，可换一个更具体的关键词（技术栈 + 场景，如 fastapi blog）。"

    lines: list[str] = []
    for i, repo in enumerate(items, 1):
        name = repo.get("full_name", "")
        desc = _bmp_text((repo.get("description") or "").replace("\n", " "), 500)
        stars = repo.get("stargazers_count", 0)
        lang = repo.get("language") or "—"
        url = repo.get("html_url", "")
        topics = ", ".join(repo.get("topics") or [])[:120]
        lines.append(
            f"{i}. {name} ({lang}, stars:{stars})\n   {desc}\n   {url}"
            + (f"\n   topics: {topics}" if topics else "")
        )
    return "\n\n".join(lines)


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
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "未安装 duckduckgo-search，无法执行网页搜索。"

    max_results = max(1, min(int(max_results), 10))
    rows: list[str] = []
    try:
        with DDGS() as ddgs:
            gen = ddgs.text(query, max_results=max_results)
            if gen is None:
                return "搜索无结果，请换关键词重试。"
            for j, item in enumerate(gen, 1):
                title = _bmp_text(item.get("title", ""), 200)
                body = _bmp_text((item.get("body") or "").replace("\n", " "), 400)
                href = item.get("href", "")
                rows.append(f"{j}. {title}\n   {body}\n   {href}")
    except Exception as e:  # noqa: BLE001
        return f"网页搜索失败: {e}"

    if not rows:
        return "搜索无结果，请换关键词或去掉 site: 限制再试。"
    return "\n\n".join(rows)
