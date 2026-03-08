#!/usr/bin/env python3
"""测试 Qwen API 响应时间和审核机制"""
import os
import time
from openai import OpenAI

api_key = os.getenv("SILICONFLOW_API_KEY", "sk-rcaanrslmsbiozibsutsmizpybekrakxhmpllqkilfepidek")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# 测试1: 简单提示词（无敏感词）
test1 = """请分析以下市场数据并给出交易建议：

收盘价: 4152.24 (涨跌幅: 0.75%)
成交量: 18,211,677,200
北向资金: 净流入 101.47 亿元

请以 JSON 格式输出：{"signal": "buy|sell|hold", "confidence": 0.0-1.0, "reasoning": "推理过程"}"""

# 测试2: 包含敏感词的提示词
test2 = """习近平主席发表重要讲话，党中央决策部署，政治局会议精神。

伊朗军事行动，伊拉克战争，中东局势紧张。

请分析市场影响并给出交易建议，以 JSON 格式输出：{"signal": "buy|sell|hold", "confidence": 0.0-1.0, "reasoning": "推理过程"}"""

# 测试3: 完整提示词（真实数据）
test3_path = "/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/docs/cache/llm_responses/llm_cache_CSI300_deepseek_v3_2.jsonl"
import json
with open(test3_path, 'r') as f:
    entry = json.loads(f.readline())
    test3 = f"""你是一位资深的量化交易分析师。请基于以下信息做出交易决策。

## 市场背景
{entry['market_context']}

## 新闻资讯
{entry['news_text'][:1000]}  # 截取前1000字符

请按照以下步骤进行分析：
1. 数据概览：总结关键信息
2. 技术分析：识别趋势和支撑/阻力位
3. 情绪评估：分析市场情绪
4. 风险识别：指出潜在风险
5. 决策输出：给出明确的交易信号

请以 JSON 格式输出：
{{
  "signal": "buy|sell|hold",
  "confidence": 0.0-1.0,
  "reasoning": "详细的推理过程"
}}"""

tests = [
    ("简单提示词（无敏感词）", test1),
    ("包含敏感词", test2),
    ("完整提示词（截取前1000字符）", test3)
]

for name, prompt in tests:
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"提示词长度: {len(prompt)} 字符")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3.5-9B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            timeout=30  # 30秒超时
        )

        latency = time.time() - start_time
        content = response.choices[0].message.content

        print(f"✅ 成功")
        print(f"延迟: {latency:.2f} 秒")
        print(f"响应长度: {len(content)} 字符")
        print(f"响应预览: {content[:200]}...")

    except Exception as e:
        latency = time.time() - start_time
        print(f"❌ 失败")
        print(f"延迟: {latency:.2f} 秒")
        print(f"错误: {str(e)}")