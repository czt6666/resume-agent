# resume-agent

基于 LangChain 1.x（`create_agent`）的简历解析与优化助手：支持上传 `.txt` / `.md` / `.pdf`，先 LLM 解析；优化阶段由**主管路由 + 多子 Agent 并行 + 汇总**完成，工具含 GitHub 与网页搜索。

## 环境

需要 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
# 在项目根创建 .env：OPENAI_API_KEY，可选 OPENAI_BASE_URL、OPENAI_MODEL、RESUME_AGENT_DATA_DIR
```

## 使用

```bash
uv run resume-agent "帮我看看简历里项目描述怎么改"
uv run resume-agent -r ./resume.md "我投后端实习，缺项目推荐"
```

## 调试

开启后会在 **stderr** 打印编排步骤（如主管规划、子专员开始、汇总），并通过 LangChain 回调打印**每次 Chat 模型调用**的完整提示词与回复摘要（前约 1200 字）。

```bash
# 任选其一
uv run resume-agent --debug -r ./resume.md
set RESUME_AGENT_DEBUG=1   # Windows PowerShell: $env:RESUME_AGENT_DEBUG=1
```

正式答案仍在 **stdout**；调试信息与最终正文分离，便于 `2> debug.log` 重定向。

## 开发

```bash
uv run python -m unittest discover -s tests -t .
```

## 目录说明


| 路径 | 说明 |
|------|------|
| `src/resume_agent/agent.py` | 主管与子 Agent 编排（单文件） |
| `src/resume_agent/tools.py` / `parser.py` / `memory.py` / `cli.py` / `debug_trace.py` | 工具、解析、持久化、命令行、调试轨迹 |
| `src/resume_agent/schemas.py` | 简历解析与 GitHub 评估的 Pydantic 模型 |
| `tests/` | 单元测试 |
| `pyproject.toml` / `uv.lock` | 依赖与锁文件 |


