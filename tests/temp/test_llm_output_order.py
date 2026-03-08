#!/usr/bin/env python3
"""
测试脚本：使用真实存在的冲突提示词验证 LLM 行为

冲突点：
- 系统提示词要求：reason → confidence → signal
- OUTPUT_FORMAT 示例：signal → confidence → reason

关键：OUTPUT_FORMAT 字段需要插入到 USER_PROMPT 中
"""

import os
import json
import requests
import time
from dotenv import load_dotenv

load_dotenv()

# ========== 旧版提示词（存在内部冲突） ==========
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

# ⚠️ 注意：这个 OUTPUT_FORMAT 的字段顺序与系统提示词要求不一致
OUTPUT_FORMAT = '{"signal":"buy"或"sell"或"hold","confidence":0.0到1.0之间的数字,"reason":"简短推理"}'

USER_PROMPT_TEMPLATE = """【决策说明】
基于以下T-1日(昨日)的数据，请做出T日(今日)的交易决策。

【T-1日市场数据】
{market_context}

【T-1日新闻事件】
{news_text}

【输出格式】
{output_format}

请严格按照系统设定的 JSON 格式给出今日的交易信号："""

# 测试数据（A 股新闻）
TEST_MARKET_CONTEXT = """### T-1日市场数据
- 日期: 2020-01-02
- 沪深300指数: 3920.51
- 涨跌幅: +1.05%
- 成交量: 8500亿元
- 波动率: 1.8%"""

TEST_NEWS = """### 1. 宏观与政策面 (CCTV)
- 央行降准0.5个百分点释放8000亿流动性：中国人民银行决定于2020年1月6日下调金融机构存款准备金率0.5个百分点，释放长期资金约8000多亿元。
- PMI指数回升至50.2，经济企稳迹象明显：2019年12月制造业PMI为50.2，连续两个月位于荣枯线以上。
- 新年戏曲晚会在京举行，党和国家领导人出席。

### 2. 成分股公告 (Notices)
- 贵州茅台：2019年净利润同比增长15%
- 中国平安：启动百亿回购计划"""

def test_with_conflict(user_prompt: str, test_name: str, model: str):
    """使用存在冲突的提示词测试 LLM"""
    print("=" * 80)
    print(f"测试: {test_name}")
    print("=" * 80)
    print("\n⚠️  提示词冲突：")
    print("  系统提示词要求：reason → confidence → signal")
    print("  OUTPUT_FORMAT：signal → confidence → reason")
    print("=" * 80)

    try:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            print("❌ 未配置 SILICONFLOW_API_KEY")
            return None

        print(f"\n调用 {model}...")

        url = "https://api.siliconflow.cn/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=120)
        latency = (time.time() - start_time) * 1000

        if response.status_code != 200:
            print(f"❌ API 错误: {response.status_code}")
            print(response.text)
            return None

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        print("\n" + "=" * 80)
        print("LLM 原始输出:")
        print("=" * 80)
        print(content)

        print("\n" + "=" * 80)
        print("延迟信息:")
        print("=" * 80)
        print(f"Latency: {latency:.1f}ms")

        # 分析字段顺序
        print("\n" + "=" * 80)
        print("字段生成顺序分析:")
        print("=" * 80)
        content_lower = content.lower()

        signal_pos = content_lower.find('"signal"')
        confidence_pos = content_lower.find('"confidence"')
        reason_pos = content_lower.find('"reason"')

        positions = []
        if signal_pos >= 0:
            positions.append(("signal", signal_pos))
        if confidence_pos >= 0:
            positions.append(("confidence", confidence_pos))
        if reason_pos >= 0:
            positions.append(("reason", reason_pos))

        positions.sort(key=lambda x: x[1])

        actual_order = ' → '.join([p[0] for p in positions])
        print(f"实际字段顺序: {actual_order}")

        # 判断遵循了哪个指令
        system_order = "reason → confidence → signal"
        output_format_order = "signal → confidence → reason"

        if actual_order == system_order:
            print(f"\n✅ LLM 遵循了系统提示词的要求：{system_order}")
            print("   LLM 先生成推理，后生成决策（正确）")
            follow = "系统提示词"
        elif actual_order == output_format_order:
            print(f"\n❌ LLM 遵循了 OUTPUT_FORMAT 的示例：{output_format_order}")
            print("   LLM 先生成决策，后生成推理（错误）")
            follow = "OUTPUT_FORMAT"
        else:
            print(f"\n⚠️  LLM 未严格遵循任一指令")
            follow = "无"

        # 尝试解析 JSON
        print("\n" + "=" * 80)
        print("JSON 解析:")
        print("=" * 80)
        try:
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            parsed = json.loads(json_str)
            print(f"✅ JSON 解析成功")
            print(f"Signal: {parsed.get('signal', 'N/A')}")
            print(f"Confidence: {parsed.get('confidence', 'N/A')}")
            print(f"Reason: {parsed.get('reason', 'N/A')}")
        except Exception as e:
            print(f"❌ JSON 解析失败: {e}")

        return {
            "actual_order": actual_order,
            "follow": follow,
            "content": content,
            "latency": latency
        }

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "=" * 80)
    print("真实冲突提示词测试")
    print("=" * 80)
    print("\n目标：验证 LLM 在冲突指令下的实际行为")
    print("冲突：系统提示词 vs OUTPUT_FORMAT")
    print("=" * 80)

    # 构建 USER_PROMPT，包含 OUTPUT_FORMAT 字段
    user_prompt = USER_PROMPT_TEMPLATE.format(
        market_context=TEST_MARKET_CONTEXT,
        news_text=TEST_NEWS,
        output_format=OUTPUT_FORMAT  # 插入 OUTPUT_FORMAT
    )

    # 测试 DeepSeek R1 8B
    print("\n\n" + "=" * 80)
    print("测试 1: DeepSeek R1 8B")
    print("=" * 80)
    result1 = test_with_conflict(
        user_prompt,
        "DeepSeek R1 8B（冲突提示词）",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    )

    # 测试 DeepSeek V3.2
    print("\n\n" + "=" * 80)
    print("测试 2: DeepSeek V3.2")
    print("=" * 80)
    result2 = test_with_conflict(
        user_prompt,
        "DeepSeek V3.2（冲突提示词）",
        "deepseek-ai/DeepSeek-V3"
    )

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    if result1:
        print(f"\nDeepSeek R1 8B:")
        print(f"  字段顺序: {result1['actual_order']}")
        print(f"  遵循指令: {result1['follow']}")

    if result2:
        print(f"\nDeepSeek V3.2:")
        print(f"  字段顺序: {result2['actual_order']}")
        print(f"  遵循指令: {result2['follow']}")

    print("\n" + "=" * 80)
    print("结论：")
    print("=" * 80)
    print("如果 LLM 遵循系统提示词（reason → confidence → signal）：")
    print("  ✅ A 股历史缓存数据可信")
    print("  ✅ 推理链条完整，不需要重新生成缓存")
    print("\n如果 LLM 遵循 OUTPUT_FORMAT（signal → confidence → reason）：")
    print("  ❌ A 股历史缓存需要重新生成")
    print("  ❌ 推理链条可能不完整")

if __name__ == "__main__":
    main()