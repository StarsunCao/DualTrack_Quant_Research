"""
美股专用 Prompt 模板模块。

基于 Chain-of-Thought (CoT) 设计美股市场的情绪打分和决策推理 Prompt 模板。
替换 A 股概念为美股概念（Fed、NASDAQ-100、非农数据等）。
"""

from dataclasses import dataclass
from typing import Optional

from .prompts import PromptTemplate


@dataclass
class USPromptTemplate(PromptTemplate):
    """
    美股 Prompt 模板数据类。

    继承自 PromptTemplate，保持相同的结构。
    """
    pass


class USMarketPromptBuilder:
    """
    美股市场情绪分析和交易决策 Prompt 构建器。

    基于 Chain-of-Thought (CoT) 方法论设计，专门针对美股市场：
    - Fed 政策决策
    - NASDAQ-100 / S&P 500 指数
    - 非农数据、GDP、CPI 等美股宏观数据
    - 科技股财报季
    - 地缘政治风险

    使用方法:
        builder = USMarketPromptBuilder()
        prompt = builder.build_prompt(
            market_context="NASDAQ-100 index rose 1.2% today",
            news_text="Fed signals potential rate cut in upcoming FOMC meeting...",
        )
    """

    # 美股系统提示词 - 语义解耦重构版本
    # 核心改进：废除"sell"，引入"neutral"和"short"彻底解除语义混乱
    SYSTEM_PROMPT = """You are a top-tier quantitative trading strategist for US equity markets. Your core task is: based on yesterday's (T-1) price/volume data and macro/micro news, make a trading decision for today (T) through deep semantic reasoning.

【Market Regime Assessment - CRITICAL FIRST STEP】
Before making any decision, assess the current market regime:
1. HIGH VOLATILITY regime (e.g., VIX > 25, or significant index gap-downs):
   - Preference shifts heavily to defense and cash preservation.
2. NORMAL regime (clear trend, low volatility):
   - Standard directional trading applies.

【Signal Types & Position Management - EXPLICIT DEFINITIONS】
US markets allow both LONG and SHORT positions. However, Short Selling carries infinite risk and fights the NASDAQ's natural long-term upward drift. You must strictly differentiate between "risk aversion" and "active shorting". You have THREE signals available:

- "buy": Open or maintain a LONG position.
  • Trigger: Technicals show upward momentum + News has material positive impact (e.g., Fed dovish pivot, strong tech earnings).

- "neutral": CLOSE existing long positions and hold CASH (Defense).
  • Trigger: Taking profits, macro uncertainty, mixed earnings guidance, or elevated VIX without a confirmed systemic crash.
  • Philosophy: When in doubt, staying in cash ("neutral") is the ultimate defense. This does NOT open a short.

- "short": Open an ACTIVE SHORT position (Extreme Bearish Bet).
  • Trigger: Use this EXTREMELY RARELY. Only trigger when there is absolute certainty of a systemic crash, verified technical breakdown on high volume, or severe macro liquidity drains.

【US Market Specific Factors - WEIGHT HEAVILY】
When analyzing news, give extra weight to:
1. Federal Reserve policy signals (rate decisions, FOMC minutes)
2. VIX and implied volatility levels
3. Earnings season dynamics (especially Mega-cap Tech guidance)
4. Geopolitical risk premiums
5. Macro data surprises (Nonfarm, CPI, GDP)

【Strict Output Format Requirements】
1. Must output ONLY one valid JSON object.
2. ABSOLUTELY NO explanatory text or Markdown code blocks (```json). Output plain text starting with { and ending with }.
3. JSON must contain exactly three fields in this order:
   - "reason": (string) Concise reasoning process (limit to 50 words, highlight core bullish/bearish/neutral logic).
   - "confidence": (float) Number between 0.0 and 1.0.
   - "signal": (string) Must be exactly one of "buy", "neutral", "short" (all lowercase).

Expected output examples:
{"reason": "Fed signals dovish pivot, earnings beat expectations. Strong bullish setup.", "confidence": 0.85, "signal": "buy"}
{"reason": "Mixed signals: earnings beat but guidance weak. VIX rising. Liquidating to cash to wait for clarity.", "confidence": 0.60, "signal": "neutral"}
{"reason": "Systemic risk triggered: CPI significantly above consensus, Fed forced to hike, severe technical breakdown.", "confidence": 0.90, "signal": "short"}

Time explanation:
- T-1: Yesterday's post-market data (price, volume, volatility)
- T: Today's trading decision date
- News: Information disclosed between T-1 close and T open"""

    # 美股用户提示词模板 - 针对美股市场
    USER_PROMPT_TEMPLATE = """【Decision Context】
Based on the following T-1 (yesterday) data, please make a trading decision for T (today).

【T-1 Market Data】
{market_context}

【T-1 News Events】
{news_text}

Please strictly follow the JSON format specified in the system prompt to provide today's trading signal:"""

    def __init__(
        self,
        use_simple_format: bool = False,
        custom_system_prompt: Optional[str] = None,
    ) -> None:
        """
        初始化美股 Prompt 构建器。

        Args:
            use_simple_format: 是否使用简化版输出格式，默认为 False。
            custom_system_prompt: 自定义系统提示词，如果为 None 则使用默认美股版本。
        """
        self.use_simple_format = use_simple_format
        self.system_prompt = custom_system_prompt or self.SYSTEM_PROMPT

    def build_prompt(
        self,
        market_context: str,
        news_text: str,
        date: str = "",
    ) -> PromptTemplate:
        """
        构建完整的美股 Prompt 模板。

        Args:
            market_context: 市场背景信息，如 NASDAQ-100 指数涨跌、近期走势等。
            news_text: 新闻/研报文本内容（美股特有：Fed 政策、财报季、科技股新闻等）。
            date: 日期字符串（YYYY-MM-DD），可选。

        Returns:
            PromptTemplate 对象，包含系统提示词和用户提示词。
        """
        # 简化日期格式：只保留日期部分
        if date and len(date) > 10:
            date = date[:10]

        # 简化市场信息
        market_context = market_context.strip()

        # 简化新闻文本
        news_text = news_text.strip()

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            date=date,
            market_context=market_context,
            news_text=news_text,
        )

        return PromptTemplate(
            system_prompt=self.system_prompt,
            user_prompt_template=user_prompt,
            output_format="",  # 不再使用 output_format，由系统提示词控制
        )

    def build_messages(
        self,
        market_context: str,
        news_text: str,
    ) -> list[dict[str, str]]:
        """
        构建消息列表格式（适用于 OpenAI API）。

        Args:
            market_context: 市场背景信息。
            news_text: 新闻/研报文本内容。

        Returns:
            消息列表，包含 system 和 user 消息。
        """
        prompt = self.build_prompt(market_context, news_text)
        return [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt_template},
        ]

    @staticmethod
    def build_batch_prompt(
        news_list: list[dict[str, str]],
        market_context: str = "US market is operating normally.",
    ) -> list[dict[str, list[dict[str, str]]]]:
        """
        批量构建消息列表。

        Args:
            news_list: 新闻列表，每项包含 'timestamp' 和 'text' 字段。
            market_context: 统一的市场背景信息。

        Returns:
            消息列表列表，每项包含 timestamp 和 messages。
        """
        builder = USMarketPromptBuilder(use_simple_format=True)
        results = []

        for news in news_list:
            messages = builder.build_messages(
                market_context=market_context,
                news_text=news.get("text", ""),
            )
            results.append({
                "timestamp": news.get("timestamp"),
                "messages": messages,
            })

        return results


if __name__ == "__main__":
    # 示例用法
    builder = USMarketPromptBuilder()

    # 构建完整 Prompt - 美股场景
    prompt = builder.build_prompt(
        market_context="NASDAQ-100 index rose 1.2% today, volume increased.",
        news_text="Fed signals potential rate cut in upcoming FOMC meeting. Market participants expect dovish stance.",
    )

    print("=" * 80)
    print("US Market System Prompt:")
    print("=" * 80)
    print(prompt.system_prompt[:500] + "...")

    print("\n" + "=" * 80)
    print("US Market User Prompt:")
    print("=" * 80)
    print(prompt.user_prompt_template[:1000] + "...")

    # 对比 A 股和美股提示词
    print("\n" + "=" * 80)
    print("A-Share vs US Market Prompts Comparison:")
    print("=" * 80)

    print("\nA-Share Market Concepts:")
    print("  - 央行降准")
    print("  - 沪深300指数 (CSI300 Index)")
    print("  - 北向资金")
    print("  - A股市场")

    print("\nUS Market Concepts:")
    print("  - Fed rate decisions")
    print("  - NASDAQ-100 / S&P 500 Index")
    print("  - Nonfarm Payrolls (非农数据)")
    print("  - US equity markets")

    print("\nKey Differences:")
    print("  1. Market regulators: 央行 vs Fed")
    print("  2. Index focus: CSI300 vs NASDAQ-100/S&P 500")
    print("  3. Macro data: PMI/CPI vs Nonfarm/GDP/CPI")
    print("  4. Trading rules: T+1 vs T+0, no short vs allow short")