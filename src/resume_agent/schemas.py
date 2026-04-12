"""简历 LLM 解析用的结构化模式（Pydantic）。"""

from __future__ import annotations

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
