# resume-agent

基于 LangChain 1.x（`create_agent`）的简历解析与优化助手：支持上传 `.txt` / `.md` / `.pdf`，先用 LLM 结构化解析，再结合工具（GitHub、网页搜索）给出可执行建议。

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
uv run resume-agent -r ./cv.pdf --parse-only
```

## 开发

```bash
uv run python -m unittest discover -s tests -t .
```

## 目录说明


| 路径                           | 说明                   |
| ---------------------------- | -------------------- |
| `src/resume_agent/`          | 可安装包：Agent、解析、加载器、工具 |
| `tests/`                     | 单元测试                 |
| `pyproject.toml` / `uv.lock` | 依赖与锁文件               |


