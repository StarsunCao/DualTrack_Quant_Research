"""
LLM Track 模块。

实现大模型代理轨道，支持本地 Ollama 和云端 API 双模式。
"""

from .prompts import PromptTemplate, SentimentPromptBuilder
from .agent import LLMTradingAgent, OllamaExecutor, DeepSeekExecutor

__all__ = [
    "PromptTemplate",
    "SentimentPromptBuilder",
    "LLMTradingAgent",
    "OllamaExecutor",
    "DeepSeekExecutor",
]