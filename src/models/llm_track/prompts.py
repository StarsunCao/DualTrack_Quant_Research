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

    # 系统提示词 - 定义角色和行为规范
    SYSTEM_PROMPT = """你是一位专业的量化交易分析师，专注于金融市场情绪分析和交易信号生成。

## 你的核心能力
1. **深度分析**: 对金融新闻进行多维度情绪分析
2. **逻辑推理**: 使用 Chain-of-Thought 方法进行逐步推理
3. **风险意识**: 始终考虑市场风险和不确定性
4. **结构化输出**: 按照指定格式输出分析结果

## 分析框架
- **宏观层面**: 政策、经济数据、国际形势
- **行业层面**: 行业动态、竞争格局、技术变革
- **公司层面**: 财务数据、管理层变动、业务发展
- **市场情绪**: 资金流向、投资者情绪、技术指标

## 决策原则
1. 保守原则: 在不确定性高时倾向于观望
2. 证据导向: 每个判断都需要明确的证据支撑
3. 风险收益平衡: 综合评估潜在收益和风险
4. 时效性考量: 考虑信息的时效性和市场反应

## 输出要求
严格按照指定的 JSON 格式输出，不要添加任何额外解释。"""

    # 用户提示词模板 - 包含 CoT 引导
    USER_PROMPT_TEMPLATE = """## 市场背景
{market_context}

## 新闻/研报内容
{news_text}

## 分析任务
请按照以下步骤进行 Chain-of-Thought 推理分析：

### 第一步：信息提取
- 提取新闻中的关键信息点
- 识别涉及的行业、公司、政策等主体
- 判断信息的重要性和时效性

### 第二步：情绪分析
- 分析新闻的情感倾向（正面/负面/中性）
- 评估对市场可能的影响方向
- 考虑市场已有预期的对比

### 第三步：影响评估
- 评估对相关资产的潜在影响程度
- 考虑影响的持续时间（短期/中期/长期）
- 分析可能的风险因素

### 第四步：决策推理
- 综合以上分析得出交易建议
- 评估决策的确信程度
- 说明主要的风险点

### 第五步：输出结果
根据分析结果，输出以下 JSON 格式：

{output_format}

请开始分析："""

    # 输出格式说明
    OUTPUT_FORMAT = """```json
{
    "signal": "buy" | "sell" | "hold",
    "confidence": 0.0-1.0,
    "reasoning": {
        "key_points": ["关键信息点1", "关键信息点2"],
        "sentiment": "positive" | "negative" | "neutral",
        "impact_analysis": "影响分析说明",
        "risk_factors": ["风险因素1", "风险因素2"],
        "time_horizon": "short_term" | "medium_term" | "long_term"
    },
    "summary": "一句话总结分析结论"
}
```"""

    # 简化版输出格式（用于批量处理）
    SIMPLE_OUTPUT_FORMAT = """```json
{
    "signal": "buy" | "sell" | "hold",
    "confidence": 0.0-1.0,
    "reasoning": "简要推理过程（100字以内）"
}
```"""

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
    ) -> PromptTemplate:
        """
        构建完整的 Prompt 模板。

        Args:
            market_context: 市场背景信息，如指数涨跌、近期走势等。
            news_text: 新闻/研报文本内容。

        Returns:
            PromptTemplate 对象，包含系统提示词和用户提示词。
        """
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
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
            # 尝试直接查找 JSON 对象
            json_pattern = r"\{[^{}]*\}"
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                json_str = matches[0]
            else:
                result["reasoning"] = "无法解析响应格式"
                return result

        try:
            parsed = json.loads(json_str)

            # 提取 signal
            signal = parsed.get("signal", "hold").lower()
            if signal in ["buy", "sell", "hold"]:
                result["signal"] = signal
            else:
                result["signal"] = "hold"

            # 提取 confidence
            confidence = parsed.get("confidence", 0.5)
            if isinstance(confidence, (int, float)):
                result["confidence"] = max(0.0, min(1.0, float(confidence)))
            else:
                result["confidence"] = 0.5

            # 提取 reasoning
            if "reasoning" in parsed:
                reasoning = parsed["reasoning"]
                if isinstance(reasoning, dict):
                    # 详细格式
                    result["reasoning"] = reasoning.get("summary", str(reasoning))
                else:
                    result["reasoning"] = str(reasoning)
            elif "summary" in parsed:
                result["reasoning"] = parsed["summary"]
            else:
                result["reasoning"] = str(parsed)

            result["parse_success"] = True

        except json.JSONDecodeError as e:
            result["reasoning"] = f"JSON 解析错误: {str(e)}"

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