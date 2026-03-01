#!/usr/bin/env python
"""
DeepSeek API 连通性测试脚本。

验证：
1. .env 文件正确加载
2. API 密钥有效
3. 可以成功调用 DeepSeek API
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def test_env_loading():
    """测试环境变量加载。"""
    print("=" * 60)
    print("  步骤 1: 检查环境变量加载")
    print("=" * 60)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if api_key:
        # 显示密钥的前后各4位，中间隐藏
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***隐藏***"
        print(f"\n  ✅ DEEPSEEK_API_KEY: {masked_key}")
    else:
        print("\n  ❌ DEEPSEEK_API_KEY: 未设置")
        return False

    if base_url:
        print(f"  ✅ DEEPSEEK_BASE_URL: {base_url}")
    else:
        print(f"  ℹ️ DEEPSEEK_BASE_URL: 未设置（将使用默认值）")

    return True


def test_api_call():
    """测试 API 调用。"""
    print("\n" + "=" * 60)
    print("  步骤 2: 测试 DeepSeek API 调用")
    print("=" * 60)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n  ❌ 无法测试：API 密钥未设置")
        return False

    try:
        import requests
        import json

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        print(f"\n  正在连接: {base_url}/chat/completions")
        print("  模型: deepseek-chat")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 发送一个简单的测试请求
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个量化交易助手。"},
                {"role": "user", "content": "请用一句话回答：当前市场状态如何？"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )

        print(f"\n  HTTP 状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = result.get("usage", {})

            print("\n  ✅ API 调用成功！")
            print(f"\n  响应内容: {content[:100]}...")
            print(f"\n  Token 使用:")
            print(f"    - Prompt Tokens: {usage.get('prompt_tokens', 0)}")
            print(f"    - Completion Tokens: {usage.get('completion_tokens', 0)}")
            print(f"    - Total Tokens: {usage.get('total_tokens', 0)}")
            return True
        else:
            error_info = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            print(f"\n  ❌ API 调用失败")
            print(f"  错误信息: {error_info}")
            return False

    except requests.exceptions.Timeout:
        print("\n  ❌ 请求超时，请检查网络连接")
        return False
    except requests.exceptions.ConnectionError:
        print("\n  ❌ 连接失败，请检查网络或代理设置")
        return False
    except Exception as e:
        print(f"\n  ❌ 发生错误: {type(e).__name__}: {e}")
        return False


def test_llm_agent():
    """测试 LLM Agent 集成。"""
    print("\n" + "=" * 60)
    print("  步骤 3: 测试 LLM Trading Agent 集成")
    print("=" * 60)

    try:
        from src.models.llm_track.agent import LLMTradingAgent

        print("\n  初始化 DeepSeek 执行器...")

        agent = LLMTradingAgent(executor_type="deepseek")

        print("  ✅ Agent 初始化成功")

        # 发送测试分析请求
        print("\n  发送测试分析请求...")

        result = agent.analyze(
            news_text="央行宣布维持利率不变，市场情绪稳定。",
            market_context="当前A股市场震荡运行，沪深300指数微涨0.2%。",
            symbol="CSI300",
        )

        print("\n  ✅ 分析请求成功！")
        print(f"\n  信号: {result.get('signal', 'N/A')}")
        print(f"  置信度: {result.get('confidence', 0):.2f}")
        print(f"  推理: {result.get('reasoning', 'N/A')[:100]}...")

        return True

    except Exception as e:
        print(f"\n  ⚠️ Agent 测试失败: {type(e).__name__}: {e}")
        print("  这可能是因为模块接口问题，但 API 配置已验证成功")
        return False


def main():
    """主测试流程。"""
    print("\n" + "=" * 60)
    print("  🔑 DeepSeek API 连通性测试")
    print("=" * 60)

    # 步骤 1: 检查环境变量
    env_ok = test_env_loading()
    if not env_ok:
        print("\n❌ 环境变量加载失败，请检查 .env 文件")
        return

    # 步骤 2: 测试 API 调用
    api_ok = test_api_call()

    # 步骤 3: 测试 Agent 集成（可选）
    if api_ok:
        test_llm_agent()

    # 最终总结
    print("\n" + "=" * 60)
    print("  📋 测试结果汇总")
    print("=" * 60)

    if api_ok:
        print("\n  ✅ DeepSeek API 配置正确，可以正常使用！")
        print("\n  使用方法:")
        print("    python main.py run --symbol CSI300  # 使用 DeepSeek 进行分析")
        print("    python main.py cache-build --executor deepseek --news-file data/news.csv")
    else:
        print("\n  ❌ API 连接失败，请检查：")
        print("    1. API 密钥是否正确")
        print("    2. 网络连接是否正常")
        print("    3. 是否需要配置代理")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()