#!/usr/bin/env python
"""
SiliconFlow API 连通性测试脚本。

测试内容：
1. 环境变量加载
2. API Key 有效性
3. 模型推理连通性
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("=" * 60)
print("SiliconFlow API 连通性测试")
print("=" * 60)

# 1. 测试环境变量
print("\n[1/4] 环境变量检查")
api_key = os.environ.get("SILICONFLOW_API_KEY")
if api_key:
    print(f"  ✓ SILICONFLOW_API_KEY 已设置: {api_key[:20]}...")
else:
    print("  ✗ SILICONFLOW_API_KEY 未设置")
    exit(1)

base_url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
print(f"  ✓ BASE_URL: {base_url}")

# 2. 测试 API 连通性
print("\n[2/4] API 服务连通性测试")
try:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30,
    )

    # 简单请求测试
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10,
    )
    print(f"  ✓ API 连接成功")
    print(f"    - 模型: {response.model}")
    print(f"    - 响应内容: {response.choices[0].message.content}")

except Exception as e:
    print(f"  ✗ API 连接失败: {e}")
    exit(1)

# 3. 测试完整推理
print("\n[3/4] 完整推理测试")
try:
    import time

    start_time = time.time()

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        messages=[
            {"role": "system", "content": "你是一个量化交易分析师。请给出买入、卖出或持有建议。"},
            {"role": "user", "content": "央行宣布降准50个基点，对股市有什么影响？请输出JSON格式：{signal: buy/sell/hold, confidence: 0-1}"},
        ],
        temperature=0.7,
        max_tokens=512,
    )

    latency_ms = (time.time() - start_time) * 1000
    content = response.choices[0].message.content
    usage = response.usage

    print(f"  ✓ 推理成功")
    print(f"    - 延迟: {latency_ms:.2f}ms")
    print(f"    - 输入 tokens: {usage.prompt_tokens}")
    print(f"    - 输出 tokens: {usage.completion_tokens}")
    print(f"    - 总 tokens: {usage.total_tokens}")
    print(f"    - 响应内容预览:\n{'-'*40}")
    print(content[:300] + "..." if len(content) > 300 else content)
    print("-" * 40)

except Exception as e:
    print(f"  ✗ 推理失败: {e}")
    exit(1)

# 4. 测试通过 LLMTradingAgent
print("\n[4/4] LLMTradingAgent 集成测试")
try:
    import sys
    sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))

    from src.models.llm_track.agent import LLMTradingAgent

    agent = LLMTradingAgent(
        executor_type="siliconflow",
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    )

    # 健康检查
    health = agent.health_check()
    print(f"  ✓ Agent 初始化成功")
    print(f"    - 执行器类型: {health['executor_type']}")
    print(f"    - 模型: {health['model']}")
    print(f"    - 健康状态: {'✓ 正常' if health['is_healthy'] else '✗ 异常'}")

    # 单次分析测试
    print(f"\n  执行单次分析...")
    result = agent.analyze(
        news_text="央行宣布降准50个基点，释放长期资金约1万亿元。",
        market_context="A股市场今日震荡上行，沪深300指数上涨0.8%。",
        symbol="CSI300",
    )

    print(f"    - 信号: {result.signal}")
    print(f"    - 置信度: {result.confidence}")
    print(f"    - 延迟: {result.latency_ms:.2f}ms")
    print(f"    - 解析成功: {result.parse_success}")

except Exception as e:
    print(f"  ✗ Agent 测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！SiliconFlow API 可用")
print("=" * 60)
