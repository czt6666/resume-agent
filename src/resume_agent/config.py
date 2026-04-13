import os

from dotenv import load_dotenv

from resume_agent.debug_trace import LlmDebugCallbackHandler, is_debug

load_dotenv()


def get_llm():
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.3,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL") or None,
    }
    if is_debug():
        kwargs["callbacks"] = [LlmDebugCallbackHandler()]
    return ChatOpenAI(**kwargs)
