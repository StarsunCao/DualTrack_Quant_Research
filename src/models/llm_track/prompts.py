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


class SmartPromptBuilder:
    """
    状态增强型 Smart Prompt 构建器。

    在 SentimentPromptBuilder 的基础上，增加对技术指标、价格序列、
    历史决策记忆的注入，使 LLM 在单次调用中做"类 Agent"的结构化推理。

    核心差异:
    - system prompt 含决策权重指引（冲突仲裁规则）
    - user prompt 额外注入技术指标摘要、近 N 日价格走势、决策记忆+反馈
    """

    SYSTEM_PROMPT = """你是一个顶级的量化交易策略分析师。你的核心任务是：基于 T-1 日的量价数据、技术指标与市场新闻，通过深度语义推理，给出 T 日的交易决策。

【核心决策逻辑：拥抱概率优势】
量化交易的核心是利用胜率和盈亏比赚钱，而非追求 100% 的确定性。请综合评估技术面与消息面，即使在多空交织的情况下，也要努力寻找天平倾斜的方向：
- buy: 技术面展现明确上攻动能/超卖反弹迹象，且消息面无重大实质性利空（允许新闻处于中性或温和）。
- sell: 技术面确认破位下行，或消息面出现明确的"黑天鹅"、系统性风险、基本面恶化。
- hold: 仅适用于极度缩量、毫无波澜的死水市场，或者多空力量处于绝对的 50:50 僵持状态。

【强制站队与决策权重】
禁止将 hold 作为逃避复杂决策的舒适区！当遇到新闻面与技术面发生冲突（多空交织）时，请遵循以下原则：
1. 黑天鹅一票否决: 仅当 VIX 极度飙升或突发严重政策收紧时，负面新闻才具有最高优先级。
2. 尊重价格行为(Price Action): 在常态震荡或常规利空利多下，请赋予【技术指标和量价走势】更高的权重。利空不跌往往是强烈的 buy 信号；利好不涨往往是 sell 信号。
3. 倾斜即站队: 如果倾向性为 55% 看多，45% 看空，请果断输出 "buy" 并将 confidence 设为 0.55，绝不允许输出 hold！

【技术指标参考】
你同时会收到技术指标摘要和近 N 日价格走势，请重点参考：
- RSI 超买超卖区、MACD 金叉死叉、均线排列方向。
- 放量下跌/缩量上涨等量价背离信号。

【历史决策反馈与进化】
参考过去的决策记录，你需要进行双向反思：
- 亏损反思：如果过去的 buy 决策导致了实际亏损，反思当时的推理是否有误。
- 踏空反思：如果过去因为过度保守输出 hold 而错过了一波上涨，你今天必须更加积极，勇于捕捉类似的技术面启动信号。

【严格输出格式要求】
1. 只能且必须输出一个合法的 JSON 对象。
2. 绝对禁止输出任何 JSON 之外的说明性文字、问候语。
3. 绝对禁止使用 ```json 和 ``` 这样的 Markdown 标记包裹结果。
4. JSON 必须严格按照以下顺序包含三个字段：
   - "reason": (字符串) 简明扼要的推理过程（限50字以内，点明最核心的多空逻辑）。
   - "confidence": (浮点数) 0.50 到 1.0 之间的数字。如果你非常犹豫，请设为 0.50 到 0.60 之间。
   - "signal": (字符串) 只能是 "buy", "sell", "hold" 三者之一（必须全小写）。

期望的完美输出示例：
{"reason": "虽有温和利空新闻，但技术面 RSI 触及严重超卖区且缩量企稳，赔率较高，博弈均值回归。", "confidence": 0.65, "signal": "buy"}

时间说明：
- T-1日：昨日收盘后的数据（价格、成交量、波动率）
- T日：今日需要做出决策的交易日
- 新闻：T-1日收盘后到T日开盘前披露的信息"""

    def __init__(
        self,
        use_simple_format: bool = False,
    ) -> None:
        self.use_simple_format = use_simple_format

    def build_messages(
        self,
        market_context: str,
        news_text: str,
        technical_summary: str = "",
        price_history: str = "",
        memory_context: str = "",
        date: str = "",
    ) -> list[dict[str, str]]:
        """
        构建增强型消息列表。

        Args:
            market_context: 市场背景（T-1日价格、VIX、北向资金等）。
            news_text: 新闻/研报文本。
            technical_summary: 技术指标摘要（自然语言格式）。
            price_history: 近 N 日价格走势（Markdown 表格）。
            memory_context: 历史决策记忆+反馈。
            date: 日期字符串。

        Returns:
            消息列表，包含 system 和 user 消息。
        """
        # 构建增强型 user prompt
        enriched_parts = []

        if date and len(date) > 10:
            date = date[:10]

        if technical_summary:
            enriched_parts.append(f"【技术指标】\n{technical_summary}")

        if price_history:
            enriched_parts.append(f"【价格走势】\n{price_history}")

        if memory_context:
            enriched_parts.append(f"【历史决策反馈】\n{memory_context}")

        if market_context:
            enriched_parts.append(f"【T-1日市场数据】\n{market_context.strip()}")

        if news_text:
            enriched_parts.append(f"【T-1日新闻事件】\n{news_text.strip()}")

        enriched_parts.append("请严格按照系统设定的 JSON 格式给出今日的交易信号：")

        user_prompt = "\n\n".join(enriched_parts)

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]


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

    # 系统提示词 - v2: 概率优势驱动，消除风险厌恶偏差
    SYSTEM_PROMPT = """你是一个顶级的量化交易策略分析师。你的核心任务是：基于昨日(T-1日)的量价数据与宏观/微观新闻，通过深度语义推理，给出今日(T日)的交易决策。

【核心决策逻辑：拥抱概率优势】
量化交易的核心是利用胜率和盈亏比赚钱，而非追求 100% 的确定性。请综合评估技术面与消息面，即使在多空交织的情况下，也要努力寻找天平倾斜的方向：
- buy: 技术面展现明确上攻动能/超卖反弹迹象，且消息面无重大实质性利空（允许新闻处于中性或温和）。
- sell: 技术面确认破位下行，或消息面出现明确的"黑天鹅"、系统性风险、基本面恶化。
- hold: 仅适用于极度缩量、毫无波澜的死水市场，或者多空力量处于绝对的 50:50 僵持状态。

【强制站队与决策权重】
禁止将 hold 作为逃避复杂决策的舒适区！当遇到新闻面与技术面发生冲突（多空交织）时，请遵循以下原则：
1. 黑天鹅一票否决: 仅当 VIX 极度飙升或突发严重政策收紧时，负面新闻才具有最高优先级。
2. 尊重价格行为(Price Action): 在常态震荡或常规利空利多下，请赋予【技术指标和量价走势】更高的权重。利空不跌往往是强烈的 buy 信号；利好不涨往往是 sell 信号。
3. 倾斜即站队: 如果倾向性为 55% 看多，45% 看空，请果断输出 "buy" 并将 confidence 设为 0.55，绝不允许输出 hold！

【历史决策反馈与进化】
参考过去的决策记录，你需要进行双向反思：
- 亏损反思：如果过去的 buy 导致了亏损，反思是否忽略了关键的技术破位或宏观隐患。
- 踏空反思：如果过去因为过度保守输出 hold 而错过了一波上涨，你今天必须更加积极，勇于捕捉类似的技术面启动信号。

【缺失信息处理原则】
如果新闻文本中未提及某些宏观数据（如CPI、资金流向等），不要自行捏造或幻觉，仅基于已知事实进行推理。

【严格输出格式要求】
1. 只能且必须输出一个合法的 JSON 对象。
2. 绝对禁止输出任何 JSON 之外的说明性文字、问候语。
3. 绝对禁止使用 ```json 和 ``` 这样的 Markdown 标记包裹结果，直接输出以 { 开头，} 结尾的纯文本。
4. JSON 必须严格按照以下顺序包含三个字段：
   - "reason": (字符串) 简明扼要的推理过程（限50字以内，点明最核心的多空逻辑）。
   - "confidence": (浮点数) 0.50 到 1.0 之间的数字。如果你非常犹豫，请设为 0.50 到 0.60 之间。
   - "signal": (字符串) 只能是 "buy", "sell", "hold" 三者之一（必须全小写）。

期望的完美输出示例：
{"reason": "虽有温和利空新闻，但技术面 RSI 触及严重超卖区且缩量企稳，赔率较高，博弈均值回归。", "confidence": 0.65, "signal": "buy"}

时间说明：
- T-1日：昨日收盘后的数据（价格、成交量、波动率）
- T日：今日需要做出决策的交易日
- 新闻：T-1日收盘后到T日开盘前披露的信息"""

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
                result["signal"] = signal if signal in ["buy", "sell", "hold", "neutral", "short"] else "hold"
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
                result["signal"] = signal if signal in ["buy", "sell", "hold", "neutral", "short"] else "hold"
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
                result["signal"] = signal if signal in ["buy", "sell", "hold", "neutral", "short"] else "hold"
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
            signal_map = {"b": "buy", "s": "sell", "h": "hold", "n": "neutral", "sh": "short"}
            signal = signal_map.get(signal, signal)

            # 验证 signal
            if signal in ["buy", "sell", "hold", "neutral", "short"]:
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