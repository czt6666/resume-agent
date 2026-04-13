"""简历优化 Agent（LangChain 1.x）——重依赖通过 __getattr__ 惰性加载，便于轻量子模块单测。"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ParsedResume": ("resume_agent.schemas", "ParsedResume"),
    "ResumeOrchestrationPlan": ("resume_agent.schemas", "ResumeOrchestrationPlan"),
    "build_agent_context": ("resume_agent.parser", "build_agent_context"),
    "build_resume_agent": ("resume_agent.agent", "build_resume_agent"),
    "load_resume_text": ("resume_agent.loaders", "load_resume_text"),
    "parse_resume_with_llm": ("resume_agent.parser", "parse_resume_with_llm"),
    "parse_then_optimize": ("resume_agent.parser", "parse_then_optimize"),
    "run_agent_turn": ("resume_agent.agent", "run_agent_turn"),
    "run_resume_orchestration": ("resume_agent.agent", "run_resume_orchestration"),
    "run_resume_pipeline_debug": ("resume_agent.orchestrator", "run_resume_pipeline_debug"),
}

__all__ = [
    "ParsedResume",
    "ResumeOrchestrationPlan",
    "build_agent_context",
    "build_resume_agent",
    "load_resume_text",
    "parse_resume_with_llm",
    "parse_then_optimize",
    "run_agent_turn",
    "run_resume_orchestration",
    "run_resume_pipeline_debug",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        mod_name, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
