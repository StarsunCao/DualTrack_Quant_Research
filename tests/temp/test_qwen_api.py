#!/usr/bin/env python3
"""测试 Qwen API 响应格式"""
import os
import json
from openai import OpenAI

# 从环境变量读取 API Key
api_key = os.getenv("SILICONFLOW_API_KEY")
if not api_key:
    print("错误：未设置 SILICONFLOW_API_KEY 环境变量")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# 测试提示词（简化版）
test_prompt = """你是一位资深的量化交易分析师。请基于以下信息做出交易决策。

## 市场背景
- 日期: 2020-01-02
- 收盘价: 4152.24 (涨跌幅: 0.75%)
- 波动率: 1.24%
- 成交量: 18,211,677,200
- 北向资金: 净流入 101.47 亿元

## 新闻资讯
央行降准释放流动性，政策环境积极。

请按照以下步骤进行分析：
1. 数据概览：总结关键信息
2. 技术分析：识别趋势和支撑/阻力位
3. 情绪评估：分析市场情绪
4. 风险识别：指出潜在风险
5. 决策输出：给出明确的交易信号

请以 JSON 格式输出：
{
  "signal": "buy|sell|hold",
  "confidence": 0.0-1.0,
  "reasoning": "详细的推理过程"
}"""

print("测试 Qwen3.5-9B API 响应格式...\n")

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-9B",
        messages=[{"role": "user", "content": test_prompt}],
        temperature=0.7,
        max_tokens=500
    )

    content = response.choices[0].message.content

    print("=" * 60)
    print("完整响应:")
    print("=" * 60)
    print(content)
    print("=" * 60)

    # 尝试解析
    print("\n尝试解析 JSON...")
    try:
        # 尝试提取 JSON 块
        import re
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, content, re.DOTALL)

        if matches:
            json_str = matches[0]
            print(f"找到 Markdown 代码块: {json_str[:100]}...")
        else:
            # 尝试直接查找 JSON 对象
            json_pattern = r"\{[^{}]*\}"
            matches = re.findall(json_pattern, content, re.DOTALL)
            if matches:
                json_str = matches[0]
                print(f"找到 JSON 对象: {json_str[:100]}...")
            else:
                print("❌ 未找到 JSON 格式")
                json_str = None

        if json_str:
            parsed = json.loads(json_str)
            print(f"\n✅ 解析成功:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ 解析失败: {e}")

except Exception as e:
    print(f"API 调用失败: {e}")