#!/usr/bin/env python3
"""分析提示词长度和可能的敏感词"""
import json
from pathlib import Path

cache_file = Path("/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/docs/cache/llm_responses/llm_cache_CSI300_deepseek_v3_2.jsonl")

with open(cache_file, 'r') as f:
    entries = [json.loads(line) for line in f]

print(f"总条目数: {len(entries)}\n")

# 分析提示词长度
news_lengths = [len(e['news_text']) for e in entries]
market_lengths = [len(e['market_context']) for e in entries]
total_lengths = [n + m for n, m in zip(news_lengths, market_lengths)]

print("提示词长度统计:")
print(f"  News 文本: 平均 {sum(news_lengths)/len(news_lengths):.0f} 字符")
print(f"  Market 背景: 平均 {sum(market_lengths)/len(market_lengths):.0f} 字符")
print(f"  总输入长度: 平均 {sum(total_lengths)/len(total_lengths):.0f} 字符")
print(f"  总输入长度: 最大 {max(total_lengths)} 字符")
print(f"  总输入长度: 最小 {min(total_lengths)} 字符")

# 检查第一个条目的详细内容
entry = entries[0]
print(f"\n第一个条目:")
print(f"  News 文本长度: {len(entry['news_text'])} 字符")
print(f"  Market 背景长度: {len(entry['market_context'])} 字符")

# 检查可能触发审核的敏感词
sensitive_keywords = [
    "习近平", "党中央", "政治局", "恐怖主义", "恐怖袭击",
    "伊朗", "伊拉克", "中东", "军事", "战争",
    "下跌", "暴跌", "崩盘", "风险", "利空"
]

print(f"\n敏感词检查 (第一个条目):")
for keyword in sensitive_keywords:
    count = entry['news_text'].count(keyword)
    if count > 0:
        print(f"  '{keyword}': {count} 次")

# 检查 Qwen 缓存
qwen_file = Path("/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/docs/cache/llm_responses/llm_cache_CSI300_qwen35_9b.jsonl")
if qwen_file.exists():
    with open(qwen_file, 'r') as f:
        qwen_entries = [json.loads(line) for line in f]

    print(f"\n\nQwen 缓存分析:")
    print(f"总条目数: {len(qwen_entries)}")

    # 检查错误类型
    error_types = {}
    for e in qwen_entries:
        reasoning = e.get('reasoning', '')
        if 'timeout' in reasoning.lower() or 'timed out' in reasoning.lower():
            error_types['timeout'] = error_types.get('timeout', 0) + 1
        elif '无法解析' in reasoning:
            error_types['parse_error'] = error_types.get('parse_error', 0) + 1
        else:
            error_types['other'] = error_types.get('other', 0) + 1

    print(f"\n错误类型分布:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")

    # 检查延迟
    latencies = [e['latency_ms'] for e in qwen_entries]
    print(f"\n延迟统计:")
    print(f"  平均: {sum(latencies)/len(latencies)/1000:.1f} 秒")
    print(f"  最小: {min(latencies)/1000:.1f} 秒")
    print(f"  最大: {max(latencies)/1000:.1f} 秒")