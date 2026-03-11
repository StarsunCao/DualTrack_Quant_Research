#!/usr/bin/env python3
"""
缓存验证脚本
验证 LLM 缓存的完整性和时间对齐

使用方法:
    uv run python scripts/validate_cache.py --cache docs/cache/llm_responses/llm_cache_QQQ_deepseek_v3_2.jsonl --ohlcv data/raw/real_qqq_5y.csv
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


def validate_cache(cache_file: Path, ohlcv_file: Path) -> list[str]:
    """
    验证缓存的完整性和时间对齐

    Args:
        cache_file: 缓存文件路径
        ohlcv_file: OHLCV 数据文件路径

    Returns:
        错误列表
    """
    print(f"验证缓存文件: {cache_file}")
    print(f"交易日数据: {ohlcv_file}")

    # 检查文件存在
    if not cache_file.exists():
        return [f"缓存文件不存在: {cache_file}"]
    if not ohlcv_file.exists():
        return [f"OHLCV文件不存在: {ohlcv_file}"]

    # 读取缓存
    records = []
    with open(cache_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"缓存记录数: {len(records)}")

    # 读取交易日
    ohlcv = pd.read_csv(ohlcv_file, parse_dates=['date'])
    ohlcv.set_index('date', inplace=True)
    trading_days = set(ohlcv.index.strftime('%Y-%m-%d'))

    errors = []
    warning_count = 0

    for r in records:
        ts = r['timestamp'][:10]
        dt = datetime.strptime(ts, '%Y-%m-%d')

        # 1. 检查是否为交易日
        if ts not in trading_days:
            errors.append(f"{ts}: 非交易日")
            continue

        # 2. 检查新闻内容
        news = r.get('news_text', '')
        if not news or not news.strip():
            errors.append(f"{ts}: 新闻内容为空")

        # 3. 检查市场背景（应包含 T-1 日期）
        market_ctx = r.get('market_context', '')
        if not market_ctx or not market_ctx.strip():
            errors.append(f"{ts}: 市场背景为空")

        # 4. 检查时间对齐：市场背景中的日期应该是 T-1
        if 'T-1日市场数据' in market_ctx:
            match = re.search(r'日期: (\d{4}-\d{2}-\d{2})', market_ctx)
            if match:
                t1_date = match.group(1)
                if t1_date >= ts:
                    errors.append(f"{ts}: T-1日期({t1_date})不在决策日之前")
            else:
                warning_count += 1
        else:
            warning_count += 1

        # 5. 检查 VIX 数据
        if 'VIX' not in market_ctx:
            warning_count += 1

        # 6. 检查必要字段
        required = ['signal', 'confidence', 'reasoning', 'model']
        for field in required:
            if field not in r:
                errors.append(f"{ts}: 缺少字段 {field}")

        # 7. 检查信号值 (支持做空信号 'short')
        signal = r.get('signal', '')
        if signal not in ['buy', 'sell', 'neutral', 'hold', 'short']:
            errors.append(f"{ts}: 无效信号值 '{signal}'")

        # 8. 检查置信度范围
        confidence = r.get('confidence', 0)
        if not (0 <= confidence <= 1):
            errors.append(f"{ts}: 置信度超出范围 {confidence}")

    print(f"\n验证完成:")
    print(f"  总记录数: {len(records)}")
    print(f"  错误数: {len(errors)}")
    print(f"  警告数: {warning_count}")

    return errors


def main():
    parser = argparse.ArgumentParser(description='验证 LLM 缓存')
    parser.add_argument('--cache', required=True, help='缓存文件路径')
    parser.add_argument('--ohlcv', required=True, help='OHLCV 数据文件路径')

    args = parser.parse_args()

    errors = validate_cache(Path(args.cache), Path(args.ohlcv))

    if errors:
        print("\n错误详情（前20条）:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors) - 20} 条错误")
        exit(1)
    else:
        print("\n✅ 缓存验证通过")
        exit(0)


if __name__ == '__main__':
    main()