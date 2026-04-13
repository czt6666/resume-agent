# resume-agent

基于 LangChain 1.x（`create_agent`）的简历解析与优化助手：支持上传 `.txt` / `.md` / `.pdf`，先 LLM 解析；优化阶段由**主管路由 + 多子 Agent 并行 + 汇总**完成，工具含 GitHub 与网页搜索。

## 环境

需要 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
cp .env.example .env
# 编辑 .env：OPENAI_API_KEY，可选 OPENAI_BASE_URL、OPENAI_MODEL
```

## 使用

```bash
uv run resume-agent "帮我看看简历里项目描述怎么改"
uv run resume-agent -r ./resume.md "我投后端实习，缺项目推荐"
```

## 开发

```bash
uv run python -m unittest discover -s tests -t .
```

## 目录说明


| 路径 | 说明 |
|------|------|
| `src/resume_agent/agent.py` | 主管与子 Agent 编排（单文件） |
| `src/resume_agent/tools.py` / `parser.py` / `memory.py` / `cli.py` | 工具、解析、持久化、命令行 |
| `src/resume_agent/schemas.py` | 简历解析与 GitHub 评估的 Pydantic 模型 |
| `tests/` | 单元测试 |
| `pyproject.toml` / `uv.lock` | 依赖与锁文件 |


