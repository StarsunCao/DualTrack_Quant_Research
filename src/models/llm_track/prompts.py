"""
Prompt 模板模块。

基于 Chain-of-Thought (CoT) 设计情绪打分和决策推理的 Prompt 模板。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """
    Prompt 模板数据类。

    Attributes:
        system_prompt: 系统提示词。
        user_prompt_template: 用户提示词模板。
        output_format: 期望的输出格式说明。
    """
    system_prompt: str
    user_prompt_template: str
    output_format: str


class SentimentPromptBuilder:
    """
    情绪分析和交易决策 Prompt 构建器。

    基于 Chain-of-Thought (CoT) 方法论设计，引导模型进行逐步推理，
    最终输出结构化的交易信号。

    使用方法:
        builder = SentimentPromptBuilder()
        prompt = builder.build_prompt(
            market_context="沪深300指数今日上涨1.2%",
            news_text="央行宣布降准50个基点...",
        )
    """

    # 系统提示词 - 优化版：强调语义推理而非规则引擎
    SYSTEM_PROMPT = """你是一个顶级的量化交易策略分析师。你的核心任务是：基于昨日(T-1日)的量价数据与宏观/微观新闻，通过深度语义推理，给出今日(T日)的交易决策。

【核心决策逻辑与风控原则】
请综合评估技术面与消息面的共振效应。你的优势在于理解新闻背后的"隐藏风险"与"情绪周期"：
- buy: 技术面展现企稳或上攻动能 + 消息面存在实质性利好（如政策强力支持、产业景气度提升、超预期财报）。
- sell (防守第一): 只要侦测到以下任意高危信号，必须果断避险：
  1. 技术面破位（如大幅下跌，特别是放量下跌）。
  2. 消息面出现"黑天鹅"或系统性风险（如地缘冲突、政策突发收紧、严重财务造假）。
  3. 宏观流动性恶化或市场情绪极度恐慌。
- hold: 消息面多空交织/方向不明，或技术面处于无趋势震荡。在"看不懂"或"不确定"时，空仓/持仓观望是最佳策略。

【缺失信息处理原则】
如果新闻文本中未提及某些宏观数据（如CPI、资金流向等），不要自行捏造或幻觉，仅基于已知事实进行推理。

【严格输出格式要求】
1. 只能且必须输出一个合法的 JSON 对象。
2. 绝对禁止输出任何 JSON 之外的说明性文字、问候语。
3. 绝对禁止使用 ```json 和 ``` 这样的 Markdown 标记包裹结果，直接输出以 { 开头，} 结尾的纯文本。
4. JSON 必须严格按照以下顺序包含三个字段：
   - "reason": (字符串) 简明扼要的推理过程（限50字以内，点明最核心的多空逻辑）。
   - "confidence": (浮点数) 0.0 到 1.0 之间的数字，代表你对该决策的置信度。
   - "signal": (字符串) 只能是 "buy", "sell", "hold" 三者之一（必须全小写）。

期望的完美输出示例：
{"reason": "T-1日放量下跌显示资金出逃，且突发行业监管收紧新闻，系统性风险加剧。", "confidence": 0.85, "signal": "sell"}

时间说明：
- T-1日：昨日收盘后的数据（价格、成交量、波动率）
- T日：今日需要做出决策的交易日
- 新闻：T-1日收盘后到T日开盘前披露的信息

风险控制原则：
- 宁可错过机会，不可承担过大风险
- 下跌趋势中果断减仓
- 重大风险事件前主动避险"""

    # 用户提示词模板 - 简洁明确
    USER_PROMPT_TEMPLATE = """【决策说明】
基于以下T-1日(昨日)的数据，请做出T日(今日)的交易决策。

【T-1日市场数据】
{market_context}

【T-1日新闻事件】
{news_text}

请严格按照系统设定的 JSON 格式给出今日的交易信号："""

    def __init__(
        self,
        use_simple_format: bool = False,
        custom_system_prompt: Optional[str] = None,
    ) -> None:
        """
        初始化 Prompt 构建器。

        Args:
            use_simple_format: 是否使用简化版输出格式，默认为 False。
            custom_system_prompt: 自定义系统提示词，如果为 None 则使用默认。
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
        构建完整的 Prompt 模板。

        Args:
            market_context: 市场背景信息，如指数涨跌、近期走势等。
            news_text: 新闻/研报文本内容。
            date: 日期字符串（YYYY-MM-DD），可选。

        Returns:
            PromptTemplate 对象，包含系统提示词和用户提示词。
        """
        # 简化日期格式：只保留日期部分
        if date and len(date) > 10:
            date = date[:10]

        # 简化市场信息（如果过长）
        market_context = market_context.strip()

        # 简化新闻文本（移除多余空格）
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
        market_context: str = "当前市场正常运行。",
    ) -> list[dict[str, list[dict[str, str]]]]:
        """
        批量构建消息列表。

        Args:
            news_list: 新闻列表，每项包含 'timestamp' 和 'text' 字段。
            market_context: 统一的市场背景信息。

        Returns:
            消息列表列表，每项包含 timestamp 和 messages。
        """
        builder = SentimentPromptBuilder(use_simple_format=True)
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


class TradingDecisionParser:
    """
    交易决策解析器。

    解析 LLM 返回的 JSON 格式交易决策。
    """

    @staticmethod
    def parse_response(response_text: str) -> dict:
        """
        解析 LLM 响应文本，提取交易决策。

        Args:
            response_text: LLM 返回的响应文本。

        Returns:
            解析后的交易决策字典，包含：
            - signal: 买卖信号 ('buy', 'sell', 'hold')
            - confidence: 确信度 (0.0-1.0)
            - reasoning: 推理过程
            - raw_response: 原始响应文本
        """
        import json
        import re

        result = {
            "signal": "hold",
            "confidence": 0.5,
            "reasoning": "",
            "raw_response": response_text,
            "parse_success": False,
        }

        # 尝试提取 JSON 块
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        if matches:
            json_str = matches[0]
        else:
            # 尝试直接查找 JSON 对象（支持紧凑格式）
            json_pattern = r"\{[^{}]*\}"
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                json_str = matches[0]
            else:
                result["reasoning"] = "无法解析响应格式"
                return result

        try:
            parsed = json.loads(json_str)

            # ========== 新增：处理错误格式 ==========

            # 格式1: {"decision": "hold"} 或 {"decision": "sell", "rationale": "..."}
            if "decision" in parsed:
                signal = parsed.get("decision", "hold").lower()
                reasoning = parsed.get("rationale", parsed.get("reasoning", ""))
                confidence = 0.6
                result["signal"] = signal if signal in ["buy", "sell", "hold"] else "hold"
                result["confidence"] = confidence
                result["reasoning"] = str(reasoning) if reasoning else "无推理说明"
                result["parse_success"] = True
                return result

            # 格式2: {"rating": "hold", "summary": "...", "stock_list": [...]}
            if "rating" in parsed and "stock_list" not in parsed:
                # 只有rating字段，直接使用
                signal = parsed.get("rating", "hold").lower()
                reasoning = parsed.get("summary", parsed.get("reason", ""))
                confidence = 0.6
                result["signal"] = signal if signal in ["buy", "sell", "hold"] else "hold"
                result["confidence"] = confidence
                result["reasoning"] = str(reasoning) if reasoning else "无推理说明"
                result["parse_success"] = True
                return result

            # 格式3: {"rating": "hold", "stock_list": [...]}
            if "rating" in parsed and "stock_list" in parsed:
                signal = parsed.get("rating", "hold").lower()
                summary = parsed.get("summary", "")

                # 提取stock_list中的关键信息
                stock_list = parsed.get("stock_list", [])
                if isinstance(stock_list, list) and len(stock_list) > 0:
                    # 提取前3条股票的reason
                    stock_reasons = []
                    for stock in stock_list[:3]:
                        if isinstance(stock, dict):
                            symbol = stock.get("symbol", "")
                            rating = stock.get("rating", "")
                            reason = stock.get("reason", "")
                            if symbol and reason:
                                stock_reasons.append(f"{symbol}({rating}): {reason}")

                    reasoning = summary
                    if stock_reasons:
                        reasoning += " | " + "; ".join(stock_reasons)
                else:
                    reasoning = summary

                confidence = 0.6  # 默认置信度
                result["signal"] = signal if signal in ["buy", "sell", "hold"] else "hold"
                result["confidence"] = confidence
                result["reasoning"] = str(reasoning) if reasoning else "无推理说明"
                result["parse_success"] = True
                return result
            # ========== 错误格式处理结束 ==========

            # 支持标准格式和紧凑格式
            # 标准格式: reason/confidence/signal (新顺序，先推理后结论)
            # 紧凑格式: r/c/s
            signal = parsed.get("signal", parsed.get("s", "hold")).lower()
            confidence = parsed.get("confidence", parsed.get("c", 0.5))
            reasoning = parsed.get("reason", parsed.get("reasoning", parsed.get("r", "")))

            # 映射紧凑信号值
            signal_map = {"b": "buy", "s": "sell", "h": "hold"}
            signal = signal_map.get(signal, signal)

            # 验证 signal
            if signal in ["buy", "sell", "hold"]:
                result["signal"] = signal
            else:
                result["signal"] = "hold"

            # 提取 confidence
            if isinstance(confidence, (int, float)):
                result["confidence"] = max(0.0, min(1.0, float(confidence)))
            else:
                result["confidence"] = 0.5

            # 提取 reasoning
            if isinstance(reasoning, dict):
                result["reasoning"] = reasoning.get("summary", str(reasoning))
            else:
                result["reasoning"] = str(reasoning) if reasoning else str(parsed)

            result["parse_success"] = True

        except json.JSONDecodeError as e:
            result["reasoning"] = f"JSON解析错误:{str(e)}"

        return result

    @staticmethod
    def signal_to_numeric(signal: str) -> int:
        """
        将信号转换为数值。

        Args:
            signal: 信号字符串 ('buy', 'sell', 'hold')。

        Returns:
            数值信号 (1, -1, 0)。
        """
        signal_map = {
            "buy": 1,
            "sell": -1,
            "hold": 0,
        }
        return signal_map.get(signal.lower(), 0)


# 预定义的 Prompt 模板示例
EXAMPLE_PROMPTS = {
    "bullish_news": """
## 市场背景
A股市场今日震荡上行，沪深300指数上涨0.8%，成交额放大。

## 新闻/研报内容
央行今日宣布下调存款准备金率50个基点，释放长期资金约1万亿元。
这是今年以来第三次降准，显示货币政策持续宽松立场。
市场分析认为，此举将有效降低银行资金成本，支持实体经济融资需求。
""",
    "bearish_news": """
## 市场背景
美股昨日大幅下跌，纳斯达克指数跌幅超过3%，科技股领跌。

## 新闻/研报内容
美联储会议纪要显示，多数委员认为通胀压力持续，可能需要维持高利率更长时间。
市场对降息预期大幅降温，风险资产面临估值压力。
多家投行下调科技股目标价，认为盈利增长面临挑战。
""",
    "neutral_news": """
## 市场背景
市场今日窄幅震荡，成交额较昨日略有萎缩。

## 新闻/研报内容
工信部发布新能源汽车产业发展规划，提出到2025年新能源汽车渗透率达到40%。
规划强调加强产业链协同发展，推动技术创新和成本降低。
分析师认为政策方向明确，但市场已有充分预期。
""",
}


if __name__ == "__main__":
    # 示例用法
    builder = SentimentPromptBuilder()

    # 构建完整 Prompt
    prompt = builder.build_prompt(
        market_context="沪深300指数今日上涨1.2%，成交额放量。",
        news_text="央行宣布降准50个基点，释放长期资金约1万亿元。",
    )

    print("=" * 60)
    print("系统提示词:")
    print("=" * 60)
    print(prompt.system_prompt[:500] + "...")

    print("\n" + "=" * 60)
    print("用户提示词:")
    print("=" * 60)
    print(prompt.user_prompt_template[:1000] + "...")

    # 测试解析器
    print("\n" + "=" * 60)
    print("解析器测试:")
    print("=" * 60)

    test_response = """
    根据分析，我认为这是一个正面信号。

    ```json
    {
        "signal": "buy",
        "confidence": 0.75,
        "reasoning": "央行降准释放流动性，对市场形成明显利好。预计短期内有上涨动力。",
        "summary": "降准利好，建议买入"
    }
    ```
    """

    parser = TradingDecisionParser()
    result = parser.parse_response(test_response)
    print(f"解析结果: {result}")