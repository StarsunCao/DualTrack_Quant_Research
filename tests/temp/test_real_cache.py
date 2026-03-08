"""测试真实LLM缓存数据的标题提取。"""

import json
import pandas as pd
from pathlib import Path

# 读取真实缓存数据
cache_file = Path("docs/cache/llm_responses/llm_cache_CSI300_siliconflow.jsonl")

if not cache_file.exists():
    print(f"缓存文件不存在: {cache_file}")
    exit(1)

# 读取前3天的缓存数据
records = []
with open(cache_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:  # 只读取前3条
            break
        if line.strip():
            records.append(json.loads(line))

print("=" * 80)
print(f"读取了 {len(records)} 条缓存记录")
print("=" * 80)

# 分析每条记录
for i, record in enumerate(records, 1):
    print(f"\n【记录 {i}】")
    print(f"时间: {record['timestamp']}")
    print(f"信号: {record['signal']}")

    # 分析news_text
    news_text = record['news_text']
    news_items = news_text.split('|')

    print(f"新闻条数: {len(news_items)}")
    print(f"\n前3条新闻:")
    for j, item in enumerate(news_items[:3], 1):
        print(f"  {j}. {item}")

    # 检查是否还有重复
    duplicates = 0
    for item in news_items:
        parts = item.split(':')
        if len(parts) >= 2:
            # 检查是否有"股票名: 股票名"的重复
            first_part = parts[0].strip()
            if first_part and len(parts) > 1 and first_part in parts[1]:
                duplicates += 1

    print(f"\n重复字段数: {duplicates}/{len(news_items)}")
    print("-" * 80)

print("\n✓ 分析完成")