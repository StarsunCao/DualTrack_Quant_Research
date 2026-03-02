#!/usr/bin/env python
"""
数据准备检查脚本。

在 fetch_complete_data.py 完成后运行，验证数据完整性。
"""

import sys
from pathlib import Path

import pandas as pd

def check_data_ready():
    """检查所有数据是否已就绪。"""
    raw_dir = Path("data/raw")

    print("=" * 60)
    print("数据准备检查")
    print("=" * 60)

    required_files = {
        "OHLCV价格": "real_csi300_5y.csv",
        "CCTV新闻": "csi300_news_cctv_2020_2024.csv",
        "A股公告": "csi300_notices_2020_2024.csv",
        "合并新闻": "csi300_news_combined_2020_2024.csv",
    }

    all_ready = True

    for name, file in required_files.items():
        path = raw_dir / file
        if path.exists():
            df = pd.read_csv(path)
            print(f"✓ {name:15s}: {len(df):5d} 条 - {file}")
        else:
            print(f"✗ {name:15s}: 文件不存在 - {file}")
            all_ready = False

    # 验证合并新闻格式
    combined_path = raw_dir / "csi300_news_combined_2020_2024.csv"
    if combined_path.exists():
        print("\n" + "-" * 60)
        df = pd.read_csv(combined_path)

        # 检查必需列
        required_cols = ["timestamp", "title", "content", "source"]
        for col in required_cols:
            if col in df.columns:
                print(f"✓ 包含列: {col}")
            else:
                print(f"✗ 缺少列: {col}")
                all_ready = False

        # 检查来源分布
        if "source" in df.columns:
            print(f"\n来源分布:")
            for source, count in df["source"].value_counts().items():
                print(f"  - {source}: {count} 条")

        # 检查日期范围
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"\n日期范围: {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")

    print("\n" + "=" * 60)
    if all_ready:
        print("✓ 所有数据已就绪，可以构建LLM缓存！")
        print("\n下一步:")
        print("  python main.py cache-build --symbol CSI300 \\")
        print("    --start 2020-01-01 --end 2024-12-31 \\")
        print("    --executor deepseek \\")
        print("    --news-file data/raw/csi300_news_combined_2020_2024.csv")
        return 0
    else:
        print("✗ 部分数据缺失，请先运行:")
        print("  python scripts/fetch_complete_data.py")
        return 1


if __name__ == "__main__":
    sys.exit(check_data_ready())
