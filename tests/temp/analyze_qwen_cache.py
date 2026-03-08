#!/usr/bin/env python3
"""分析 Qwen 缓存文件"""
import json
from pathlib import Path

cache_file = Path("/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/docs/cache/llm_responses/llm_cache_CSI300_qwen35_9b.jsonl")

with open(cache_file, 'r') as f:
    entries = [json.loads(line) for line in f]

print(f"总条目数: {len(entries)}")

# 统计推理类型
reasoning_types = {}
for entry in entries:
    reasoning = entry.get('reasoning', '')
    key = reasoning[:50] if reasoning else 'empty'
    reasoning_types[key] = reasoning_types.get(key, 0) + 1

print("\n推理类型统计:")
for key, count in sorted(reasoning_types.items(), key=lambda x: -x[1]):
    print(f"  {key}: {count}")

# 查找有 raw_response 的条目
entries_with_raw = [e for e in entries if e.get('raw_response') and e['raw_response'].strip()]
print(f"\n有 raw_response 的条目: {len(entries_with_raw)}")

if entries_with_raw:
    print("\n第一个有 raw_response 的条目:")
    entry = entries_with_raw[0]
    print(f"Signal: {entry['signal']}")
    print(f"Confidence: {entry['confidence']}")
    print(f"Reasoning: {entry['reasoning']}")
    print(f"\nRaw response (前500字符):")
    print(entry['raw_response'][:500])
    print("...")