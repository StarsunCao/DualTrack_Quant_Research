"""
LLM Track 模块。

实现大模型代理轨道，支持本地 Ollama 和云端 API 双模式。
"""

from .prompts import PromptTemplate, SentimentPromptBuilder, SmartPromptBuilder
from .us_prompts import USMarketPromptBuilder, USSmartPromptBuilder
from .agent import (
    LLMTradingAgent,
    SmartPromptAgent,
    OllamaExecutor,
    DeepSeekExecutor,
)
from .memory import DecisionMemoryStore, DecisionRecord

__all__ = [
    "PromptTemplate",
    "SentimentPromptBuilder",
    "SmartPromptBuilder",
    "USMarketPromptBuilder",
    "USSmartPromptBuilder",
    "LLMTradingAgent",
    "SmartPromptAgent",
    "OllamaExecutor",
    "DeepSeekExecutor",
    "DecisionMemoryStore",
    "DecisionRecord",
]
