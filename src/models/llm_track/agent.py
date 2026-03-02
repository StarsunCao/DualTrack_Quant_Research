"""
LLM 交易代理模块。

实现大模型代理轨道，支持本地 Ollama 和云端 API 双模式。
包含离线推理架构，支持批量处理和缓存。
"""

import json
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import requests
from tqdm import tqdm

from src.utils.logger import get_logger
from .prompts import SentimentPromptBuilder, TradingDecisionParser

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """
    LLM 响应数据类。

    Attributes:
        signal: 交易信号 ('buy', 'sell', 'hold')。
        confidence: 确信度 (0.0-1.0)。
        reasoning: 推理过程。
        latency_ms: 响应延迟（毫秒）。
        raw_response: 原始响应文本。
        parse_success: 是否成功解析。
        timestamp: 时间戳。
        symbol: 资产代码。
        model: 使用的模型名称。
        tps: Token 生成速度 (tokens/s)。
        eval_count: 生成的 token 数量。
        eval_duration_ms: 生成耗时（毫秒）。
    """
    signal: str
    confidence: float
    reasoning: str
    latency_ms: float
    raw_response: str
    parse_success: bool
    timestamp: Optional[datetime] = None
    symbol: str = "UNKNOWN"
    model: str = "unknown"
    tps: float = 0.0
    eval_count: int = 0
    eval_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "parse_success": self.parse_success,
            "tps": self.tps,
            "eval_count": self.eval_count,
            "eval_duration_ms": self.eval_duration_ms,
        }


class BaseExecutor(ABC):
    """
    LLM 执行器抽象基类。

    定义执行器的标准接口。
    """

    def __init__(self, model: str, timeout: int = 60) -> None:
        """
        初始化执行器。

        Args:
            model: 模型名称。
            timeout: 请求超时时间（秒）。
        """
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def execute(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        执行 LLM 请求。

        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]。

        Returns:
            LLMResponse 对象。
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        检查执行器是否可用。

        Returns:
            是否可用。
        """
        pass


class OllamaExecutor(BaseExecutor):
    """
    Ollama 本地执行器。

    调用本地 Ollama 接口，默认地址为 http://localhost:11434。
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.7,
    ) -> None:
        """
        初始化 Ollama 执行器。

        Args:
            model: 模型名称，默认为 qwen2.5:7b。
            base_url: Ollama 服务地址。
            timeout: 请求超时时间（秒）。
            temperature: 生成温度。
        """
        super().__init__(model=model, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def execute(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        执行 Ollama 请求。

        Args:
            messages: 消息列表。

        Returns:
            LLMResponse 对象。
        """
        start_time = time.time()

        # Ollama API 格式
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000
            result = response.json()

            # 提取响应内容
            content = result.get("message", {}).get("content", "")

            # 提取 TPS 信息（Ollama 特有字段）
            eval_count = result.get("eval_count", 0)  # 生成的 token 数
            eval_duration_ns = result.get("eval_duration", 0)  # 纳秒

            # 计算 TPS
            if eval_duration_ns > 0 and eval_count > 0:
                eval_duration_ms = eval_duration_ns / 1e6  # 纳秒转毫秒
                eval_duration_s = eval_duration_ns / 1e9  # 转为秒
                tps = eval_count / eval_duration_s
                # 打印日志
                print(f"LLM Inference: Latency={latency_ms/1000:.2f}s | Speed={tps:.1f} tokens/s | Tokens={eval_count}")
            else:
                tps = 0.0
                eval_duration_ms = 0.0

            # 解析交易决策
            parsed = TradingDecisionParser.parse_response(content)

            return LLMResponse(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                latency_ms=latency_ms,
                raw_response=content,
                parse_success=parsed["parse_success"],
                model=self.model,
                tps=tps,
                eval_count=eval_count,
                eval_duration_ms=eval_duration_ms,
            )

        except requests.exceptions.Timeout:
            return LLMResponse(
                signal="hold",
                confidence=0.0,
                reasoning="请求超时",
                latency_ms=(time.time() - start_time) * 1000,
                raw_response="",
                parse_success=False,
                model=self.model,
            )
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                signal="hold",
                confidence=0.0,
                reasoning=f"请求错误: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                raw_response="",
                parse_success=False,
                model=self.model,
            )
        except Exception as e:
            return LLMResponse(
                signal="hold",
                confidence=0.0,
                reasoning=f"未知错误: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                raw_response="",
                parse_success=False,
                model=self.model,
            )

    def health_check(self) -> bool:
        """
        检查 Ollama 服务是否可用。

        Returns:
            是否可用。
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """
        列出可用的模型。

        Returns:
            模型名称列表。
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            pass
        return []


class DeepSeekExecutor(BaseExecutor):
    """
    DeepSeek API 执行器。

    使用 OpenAI SDK 兼容模式调用 DeepSeek API。
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        timeout: int = 60,
        temperature: float = 0.7,
    ) -> None:
        """
        初始化 DeepSeek 执行器。

        Args:
            model: 模型名称，默认为 deepseek-chat。
            api_key: DeepSeek API Key，如未提供则从环境变量读取。
            base_url: API 基础 URL。
            timeout: 请求超时时间（秒）。
            temperature: 生成温度。
        """
        super().__init__(model=model, timeout=timeout)
        self.base_url = base_url
        self.temperature = temperature

        # API Key 处理
        if api_key is None:
            import os
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_key = api_key

        # 延迟导入 OpenAI（避免未安装时报错）
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端。"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError("请安装 openai 库: uv add openai")
        return self._client

    def execute(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        执行 DeepSeek API 请求。

        Args:
            messages: 消息列表。

        Returns:
            LLMResponse 对象。
        """
        start_time = time.time()

        if not self.api_key:
            return LLMResponse(
                signal="hold",
                confidence=0.0,
                reasoning="未配置 DEEPSEEK_API_KEY",
                latency_ms=0,
                raw_response="",
                parse_success=False,
                model=self.model,
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", 1024),
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""

            # 提取 TPS 信息（OpenAI SDK 格式）
            usage = response.usage
            if usage:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens  # 相当于 eval_count

                # 估算 TPS（OpenAI 不提供生成耗时，使用总延迟估算）
                total_latency_s = latency_ms / 1000
                if total_latency_s > 0 and completion_tokens > 0:
                    tps = completion_tokens / total_latency_s
                    eval_duration_ms = latency_ms
                    # 打印日志
                    print(f"DeepSeek Inference: Latency={latency_ms/1000:.2f}s | Speed={tps:.1f} tokens/s | Tokens={completion_tokens}")
                else:
                    tps = 0.0
                    eval_duration_ms = 0.0
            else:
                completion_tokens = 0
                tps = 0.0
                eval_duration_ms = 0.0

            # 解析交易决策
            parsed = TradingDecisionParser.parse_response(content)

            return LLMResponse(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                latency_ms=latency_ms,
                raw_response=content,
                parse_success=parsed["parse_success"],
                model=self.model,
                tps=tps,
                eval_count=completion_tokens,
                eval_duration_ms=eval_duration_ms,
            )

        except Exception as e:
            return LLMResponse(
                signal="hold",
                confidence=0.0,
                reasoning=f"API 错误: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                raw_response="",
                parse_success=False,
                model=self.model,
            )

    def health_check(self) -> bool:
        """
        检查 DeepSeek API 是否可用。

        Returns:
            是否可用。
        """
        if not self.api_key:
            return False
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
            )
            return bool(response.choices)
        except Exception:
            return False


class MockExecutor(BaseExecutor):
    """
    模拟执行器（用于测试和回测）。

    不调用实际 API，直接返回预设响应。
    """

    def __init__(
        self,
        model: str = "mock-model",
        default_signal: str = "hold",
        latency_ms: float = 10.0,
    ) -> None:
        """
        初始化模拟执行器。

        Args:
            model: 模型名称。
            default_signal: 默认信号。
            latency_ms: 模拟延迟（毫秒）。
        """
        super().__init__(model=model)
        self.default_signal = default_signal
        self.latency_ms = latency_ms

    def execute(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """执行模拟请求。"""
        time.sleep(self.latency_ms / 1000)  # 模拟延迟

        # 简单的关键词匹配
        content = str(messages[-1].get("content", ""))

        if any(word in content for word in ["利好", "上涨", "突破", "增长"]):
            signal, confidence = "buy", 0.7
        elif any(word in content for word in ["利空", "下跌", "风险", "压力"]):
            signal, confidence = "sell", 0.65
        else:
            signal, confidence = "hold", 0.5

        return LLMResponse(
            signal=signal,
            confidence=confidence,
            reasoning="[Mock] 基于关键词的模拟推理",
            latency_ms=self.latency_ms,
            raw_response=f'{{"signal": "{signal}", "confidence": {confidence}}}',
            parse_success=True,
            model=self.model,
        )

    def health_check(self) -> bool:
        return True


@dataclass
class CacheEntry:
    """
    缓存条目数据类。

    用于离线推理架构的 JSONL 缓存。
    """
    timestamp: str
    symbol: str
    news_text: str
    market_context: str
    signal: str
    confidence: float
    reasoning: str
    latency_ms: float
    model: str
    cache_time: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_jsonl(self) -> str:
        """转换为 JSONL 格式字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "news_text": self.news_text,
            "market_context": self.market_context,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "cache_time": self.cache_time,
        }


class LLMTradingAgent:
    """
    LLM 交易代理类。

    支持本地 Ollama 和云端 API 双模式，包含离线推理架构，
    支持批量处理和缓存。

    使用方法:
        # 本地模式
        agent = LLMTradingAgent(executor_type="ollama", model="qwen2.5:7b")

        # 云端模式
        agent = LLMTradingAgent(
            executor_type="deepseek",
            api_key="your-api-key",
        )

        # 批量推理（离线模式）
        results = agent.batch_analyze(news_list, cache_path="cache.jsonl")
    """

    def __init__(
        self,
        executor_type: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        temperature: float = 0.7,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        初始化 LLM 交易代理。

        Args:
            executor_type: 执行器类型，可选 'ollama', 'deepseek', 'mock'。
            model: 模型名称。
            api_key: API Key（用于云端模式）。
            base_url: 基础 URL（用于自定义端点）。
            timeout: 请求超时时间（秒）。
            temperature: 生成温度。
            use_cache: 是否启用缓存。
            cache_dir: 缓存目录，默认为 data/llm_cache/。
        """
        self.executor_type = executor_type
        self.use_cache = use_cache
        self.prompt_builder = SentimentPromptBuilder(use_simple_format=True)

        # 设置缓存目录
        project_root = Path(__file__).parent.parent.parent
        self.cache_dir = cache_dir or project_root / "data" / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化执行器
        self.executor = self._create_executor(
            executor_type=executor_type,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
        )

        # 缓存字典（内存缓存）
        self._cache: dict[str, CacheEntry] = {}

    def _create_executor(
        self,
        executor_type: str,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: int,
        temperature: float,
    ) -> BaseExecutor:
        """
        创建执行器实例。

        Args:
            executor_type: 执行器类型。
            model: 模型名称。
            api_key: API Key。
            base_url: 基础 URL。
            timeout: 超时时间。
            temperature: 生成温度。

        Returns:
            执行器实例。
        """
        if executor_type == "ollama":
            return OllamaExecutor(
                model=model or "qwen2.5:7b",
                base_url=base_url or "http://localhost:11434",
                timeout=timeout,
                temperature=temperature,
            )
        elif executor_type == "deepseek":
            return DeepSeekExecutor(
                model=model or "deepseek-chat",
                api_key=api_key,
                base_url=base_url or "https://api.deepseek.com/v1",
                timeout=timeout,
                temperature=temperature,
            )
        elif executor_type == "siliconflow":
            # SiliconFlow 使用 OpenAI 兼容格式，与 DeepSeek 相同
            # API Key 从 SILICONFLOW_API_KEY 环境变量读取
            if api_key is None:
                api_key = os.environ.get("SILICONFLOW_API_KEY", "")
            return DeepSeekExecutor(
                model=model or "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                api_key=api_key,
                base_url=base_url or "https://api.siliconflow.cn/v1",
                timeout=timeout,
                temperature=temperature,
            )
        elif executor_type == "mock":
            return MockExecutor(model=model or "mock-model")
        else:
            raise ValueError(f"不支持的执行器类型: {executor_type}")

    def analyze(
        self,
        news_text: str,
        market_context: str = "当前市场正常运行。",
        symbol: str = "UNKNOWN",
        timestamp: Optional[datetime] = None,
    ) -> LLMResponse:
        """
        分析单条新闻，生成交易信号。

        Args:
            news_text: 新闻文本。
            market_context: 市场背景。
            symbol: 资产代码。
            timestamp: 时间戳。

        Returns:
            LLMResponse 对象。
        """
        # 构建缓存键（优先使用时间戳，更稳定）
        if timestamp is not None:
            ts_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
            cache_key = f"{symbol}_{ts_str}"
        else:
            cache_key = f"{symbol}_{hash(news_text)}"

        # 检查内存缓存
        if self.use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return LLMResponse(
                signal=cached.signal,
                confidence=cached.confidence,
                reasoning=cached.reasoning,
                latency_ms=0,  # 缓存命中，延迟为 0
                raw_response="[CACHE HIT]",
                parse_success=True,
                timestamp=timestamp,
                symbol=symbol,
                model=cached.model,
            )

        # 构建 Prompt
        messages = self.prompt_builder.build_messages(
            market_context=market_context,
            news_text=news_text,
        )

        # 执行推理
        response = self.executor.execute(messages)
        response.timestamp = timestamp
        response.symbol = symbol

        # 更新缓存
        if self.use_cache:
            self._cache[cache_key] = CacheEntry(
                timestamp=timestamp.isoformat() if timestamp else "",
                symbol=symbol,
                news_text=news_text,
                market_context=market_context,
                signal=response.signal,
                confidence=response.confidence,
                reasoning=response.reasoning,
                latency_ms=response.latency_ms,
                model=response.model,
            )

        return response

    def batch_analyze(
        self,
        news_list: list[dict[str, Any]],
        market_context: str = "当前市场正常运行。",
        symbol: str = "UNKNOWN",
        max_workers: int = 4,
        cache_path: Optional[Path] = None,
        use_parallel: bool = True,
    ) -> pd.DataFrame:
        """
        批量分析新闻（离线推理模式）。

        支持并行请求和缓存持久化，适用于回测场景。

        Args:
            news_list: 新闻列表，每项包含 'timestamp' 和 'text' 字段。
            market_context: 市场背景。
            symbol: 资产代码。
            max_workers: 最大并行工作线程数。
            cache_path: 缓存文件路径，如提供则保存到 JSONL 文件。
            use_parallel: 是否使用并行处理。

        Returns:
            包含交易信号的 DataFrame，列包括：
            - timestamp: 时间戳
            - symbol: 资产代码
            - llm_signal: 交易信号
            - reasoning: 推理过程
            - latency_ms: 延迟
            - confidence: 确信度
        """
        results: list[dict] = []

        # 加载已有缓存
        if cache_path and cache_path.exists():
            self._load_cache(cache_path)

        def process_news(news: dict) -> dict:
            """处理单条新闻。"""
            timestamp = news.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            # 支持聚合新闻格式（每日多条新闻合并）
            # 使用包含完整内容的聚合内容（aggregated_content包含标题+内容）
            news_text = news.get(
                "aggregated_content",  # 完整聚合内容（标题+内容）
                news.get(
                    "structured_summary",  # 备用：结构化的分类摘要（仅标题）
                    news.get("text", news.get("content", ""))
                )
            )

            response = self.analyze(
                news_text=news_text,
                market_context=market_context,
                symbol=symbol,
                timestamp=timestamp,
            )

            result = {
                "timestamp": timestamp,
                "symbol": symbol,
                "llm_signal": response.signal,
                "reasoning": response.reasoning,
                "latency_ms": response.latency_ms,
                "confidence": response.confidence,
                "model": response.model,
                "parse_success": response.parse_success,
            }

            # 实时追加保存（支持断点续传）
            if cache_path:
                self._append_to_cache(cache_path, result, news_text=news_text, market_context=market_context)

            return result

        # 执行批量处理（带进度条）
        logger.info(f"开始批量分析 {len(news_list)} 条新闻...")

        # 添加速率限制参数（避免API超时）
        request_delay = 2.0  # 每次请求间隔2秒，避免速率限制

        if use_parallel and len(news_list) > 1:
            # 并行模式：减少并发数并添加延迟
            with ThreadPoolExecutor(max_workers=min(max_workers, 2)) as executor:  # 限制最多2个并发
                futures = {executor.submit(process_news, news): news for news in news_list}
                progress_bar = tqdm(total=len(news_list), desc="LLM Batch Analysis", unit="news")
                for future in as_completed(futures):
                    results.append(future.result())
                    progress_bar.update(1)
                    time.sleep(request_delay)  # 每次请求后延迟
                progress_bar.close()
        else:
            # 串行模式：添加请求间隔
            for news in tqdm(news_list, desc="LLM Analysis", unit="news"):
                results.append(process_news(news))
                time.sleep(request_delay)  # 每次请求后延迟

        # 排序结果（按时间戳）
        results.sort(key=lambda x: x["timestamp"] if x["timestamp"] else datetime.min)

        # 保存缓存（现在不需要了，因为已实时保存，但保留以防兼容性问题）
        if cache_path and len(results) > 0:
            logger.info(f"缓存已实时保存到: {cache_path}")

        # 创建 DataFrame
        df = pd.DataFrame(results)
        return df

    def _load_cache(self, cache_path: Path) -> None:
        """
        从 JSONL 文件加载缓存（支持损坏恢复）。

        Args:
            cache_path: 缓存文件路径。
        """
        try:
            corrupted_lines = 0
            loaded_count = 0

            with open(cache_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            # 使用时间戳作为缓存键（更稳定）
                            ts = entry.get("timestamp", "")
                            symbol = entry.get("symbol", "UNKNOWN")
                            if ts:
                                cache_key = f"{symbol}_{ts}"
                            else:
                                # 兼容旧缓存格式
                                cache_key = f"{symbol}_{hash(entry.get('news_text', ''))}"

                            self._cache[cache_key] = CacheEntry(**entry)
                            loaded_count += 1
                        except json.JSONDecodeError as e:
                            corrupted_lines += 1
                            print(f"⚠️ 缓存文件第 {line_num} 行损坏: {e}")
                        except KeyError as e:
                            corrupted_lines += 1
                            print(f"⚠️ 缓存文件第 {line_num} 行缺少必需字段: {e}")

            print(f"已加载 {loaded_count} 条缓存")
            if corrupted_lines > 0:
                print(f"⚠️ 共跳过 {corrupted_lines} 行损坏缓存")

        except Exception as e:
            print(f"加载缓存失败: {e}")

    def _save_cache(self, cache_path: Path, results: list[dict]) -> None:
        """
        保存缓存到 JSONL 文件（支持并发安全）。

        Args:
            cache_path: 缓存文件路径。
            results: 结果列表。
        """
        try:
            # 确保目录存在
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入 JSONL（使用文件锁确保并发安全）
            import fcntl

            with open(cache_path, "w", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他锁
                try:
                    for cache_entry in self._cache.values():
                        f.write(cache_entry.to_jsonl() + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.info(f"已保存 {len(self._cache)} 条缓存到 {cache_path}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _append_to_cache(self, cache_path: Path, entry: dict, news_text: str = "", market_context: str = "") -> None:
        """
        追加单条记录到缓存文件（实时保存，支持断点续传）。

        Args:
            cache_path: 缓存文件路径。
            entry: 要追加的缓存条目（来自 process_news 的结果字典）。
            news_text: 新闻原文（用于构建完整 CacheEntry）。
            market_context: 市场背景（用于构建完整 CacheEntry）。
        """
        try:
            # 确保目录存在
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            import fcntl

            # 处理 timestamp 格式（支持 pandas Timestamp 和 datetime）
            ts = entry.get("timestamp", "")
            if hasattr(ts, 'isoformat'):
                # pandas Timestamp 或 datetime 对象
                ts = ts.isoformat()
            elif ts is None:
                ts = ""

            # 将 entry 转换为 CacheEntry 格式
            # entry 格式: {timestamp, symbol, llm_signal, reasoning, latency_ms, confidence, model, parse_success}
            cache_entry = CacheEntry(
                timestamp=ts,
                symbol=entry.get("symbol", "UNKNOWN"),
                news_text=news_text,
                market_context=market_context,
                signal=entry.get("llm_signal", "hold"),
                confidence=entry.get("confidence", 0.0),
                reasoning=entry.get("reasoning", ""),
                latency_ms=entry.get("latency_ms", 0.0),
                model=entry.get("model", "unknown"),
            )

            # 追加模式写入
            with open(cache_path, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(cache_entry.to_jsonl() + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.warning(f"追加缓存失败: {e}")

    def health_check(self) -> dict:
        """
        检查执行器健康状态。

        Returns:
            包含状态信息的字典。
        """
        is_healthy = self.executor.health_check()
        return {
            "executor_type": self.executor_type,
            "model": self.executor.model,
            "is_healthy": is_healthy,
            "cache_enabled": self.use_cache,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """清空内存缓存。"""
        self._cache.clear()

    def get_signal_dataframe(
        self,
        news_list: list[dict[str, Any]],
        market_context: str = "当前市场正常运行。",
        symbol: str = "UNKNOWN",
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取标准化的信号 DataFrame。

        这是 batch_analyze 的别名，返回标准格式输出。

        Args:
            news_list: 新闻列表。
            market_context: 市场背景。
            symbol: 资产代码。
            **kwargs: 其他参数传递给 batch_analyze。

        Returns:
            标准格式的信号 DataFrame。
        """
        df = self.batch_analyze(
            news_list=news_list,
            market_context=market_context,
            symbol=symbol,
            **kwargs,
        )

        # 确保列顺序符合标准
        standard_columns = [
            "timestamp", "symbol", "llm_signal", "reasoning", "latency_ms",
            "confidence", "model", "parse_success",
        ]
        df = df[[col for col in standard_columns if col in df.columns]]

        return df


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("LLM Trading Agent 示例")
    print("=" * 60)

    # 使用 Mock 执行器测试
    agent = LLMTradingAgent(executor_type="mock")

    # 检查健康状态
    health = agent.health_check()
    print(f"\n执行器状态: {health}")

    # 单条分析
    print("\n单条新闻分析:")
    response = agent.analyze(
        news_text="央行宣布降准50个基点，释放长期资金约1万亿元。",
        market_context="A股市场今日震荡上行，沪深300指数上涨0.8%。",
        symbol="CSI300",
    )
    print(f"信号: {response.signal}")
    print(f"确信度: {response.confidence}")
    print(f"推理: {response.reasoning}")
    print(f"延迟: {response.latency_ms:.2f}ms")

    # 批量分析
    print("\n批量新闻分析:")
    news_list = [
        {"timestamp": "2024-01-01 10:00:00", "text": "央行降准，市场迎来利好。"},
        {"timestamp": "2024-01-02 14:00:00", "text": "美联储暗示加息，市场承压。"},
        {"timestamp": "2024-01-03 09:30:00", "text": "经济数据平稳，市场观望。"},
    ]

    results_df = agent.batch_analyze(
        news_list=news_list,
        market_context="市场整体平稳运行。",
        symbol="CSI300",
    )
    print(results_df.to_string())