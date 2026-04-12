"""简历 LLM 解析用的结构化模式（Pydantic）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EducationItem(BaseModel):
    school: str | None = None
    degree_major: str | None = None
    period: str | None = None
    highlights: list[str] = Field(default_factory=list)


class WorkItem(BaseModel):
    company: str | None = None
    role: str | None = None
    period: str | None = None
    bullets: list[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str | None = None
    description: str | None = None
    stack: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    """LLM 从原始简历文本中提取的结构化信息。"""

    name: str | None = None
    contact_masked: str | None = Field(
        None,
        description="联系方式是否出现及类型（如：有邮箱/有手机），勿输出完整号码或邮箱原文",
    )
    target_role: str | None = Field(None, description="求职意向岗位，若文中未写则 null")
    summary: str | None = Field(None, description="个人总结/自我评价要点")
    skills: list[str] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    work_experience: list[WorkItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    gaps_or_notes: list[str] = Field(
        default_factory=list,
        description="可观察到的短板：如缺实习、项目偏少、缺少量化指标、技术栈与目标岗不匹配等",
    )


class GithubRepoFitRow(BaseModel):
    """单条仓库与简历练手目标的匹配评估。"""

    repo_ref: str = Field(description="仓库 full_name，或候选列表中的序号")
    verdict: Literal["too_easy", "ok", "too_hard", "unclear"] = Field(
        description="too_easy=玩具/清单/纯 demo；ok=体量与难度适中可写简历；too_hard=需长期投入或领域过深；unclear=信息不足",
    )
    difficulty_1_10: int = Field(ge=1, le=10, description="技术深度：1 极浅，10 专家级")
    resume_value_1_10: int = Field(ge=1, le=10, description="写进简历的性价比与可信度")
    reason: str = Field(description="一两句中文理由")


class GithubFitEvaluation(BaseModel):
    """一批 GitHub 候选的整体评估与后续搜索建议。"""

    rows: list[GithubRepoFitRow] = Field(description="对每条可见候选的评估，与输入顺序或 repo 对应")
    summary: str = Field(description="整体中文小结：当前这批是否值得跟、主要问题")
    next_search_queries: list[str] = Field(
        default_factory=list,
        description="若整体不合适，给出 1～3 条新的 GitHub 搜索 query（可含 language:、stars:..、-topic:awesome 等）",
        max_length=4,
    )
