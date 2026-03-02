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

    # 系统提示词 - 增强约束力，确保格式正确
    SYSTEM_PROMPT = """量化分析师。基于每日新闻给出交易决策。

严格要求：
1. 只输出JSON格式，不要输出其他任何文字
2. 必须包含三个字段：signal, confidence, reason
3. signal只能是：buy, sell, hold（小写）
4. confidence范围：0.0-1.0之间的数字
5. reason是简短的推理文字

禁止输出：decision, rating, stock_list等字段
禁止添加任何说明文字或markdown标记"""

    # 用户提示词模板 - 强调格式
    USER_PROMPT_TEMPLATE = """{date}|{market_context}|{news_text}

严格按照此格式输出（不要修改字段名）：
{output_format}

记住：只输出JSON，不要其他文字！"""

    # 输出格式说明（更明确）
    OUTPUT_FORMAT = '''{"signal":"buy"或"sell"或"hold","confidence":0.0到1.0之间的数字,"reason":"简短推理"}'''

    # 简化版输出格式（默认）
    SIMPLE_OUTPUT_FORMAT = '{"s":"b"或"s"或"h","c":0.0到1.0,"r":"推理"}'

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
        self.output_format = (
            self.SIMPLE_OUTPUT_FORMAT if use_simple_format else self.OUTPUT_FORMAT
        )

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
            output_format=self.output_format,
        )

        return PromptTemplate(
            system_prompt=self.system_prompt,
            user_prompt_template=user_prompt,
            output_format=self.output_format,
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
            # 标准格式: signal/confidence/reasoning
            # 紧凑格式: s/c/r
            signal = parsed.get("signal", parsed.get("s", "hold")).lower()
            confidence = parsed.get("confidence", parsed.get("c", 0.5))
            reasoning = parsed.get("reasoning", parsed.get("reason", parsed.get("r", "")))

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