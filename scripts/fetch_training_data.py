#!/usr/bin/env python
"""
获取CSI300历史数据用于ML训练。

训练集：2015-01-01 到 2019-12-31（5年历史数据）
测试集：2020-01-01 到 2024-12-31（5年测试数据）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.market_data import MarketDataFetcher
import pandas as pd


def fetch_training_data():
    """获取2015-2019的训练数据。"""
    fetcher = MarketDataFetcher()

    print("=" * 70)
    print("  获取CSI300历史数据（训练集：2015-2019）")
    print("=" * 70)

    # 获取2015-2019数据
    print("\n正在获取2015-2019数据...")
    train_data = fetcher.fetch_csi300(
        start_date="2015-01-01",
        end_date="2019-12-31",
        save_to_file=False
    )

    print(f"训练数据: {len(train_data)} 天")
    print(f"时间范围: {train_data.index[0]} to {train_data.index[-1]}")

    # 保存训练数据
    train_path = Path("data/raw/csi300_train_2015_2019.csv")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path)
    print(f"✅ 训练数据已保存: {train_path}")

    # 检查测试数据
    test_path = Path("data/raw/real_csi300_5y.csv")
    if test_path.exists():
        test_data = pd.read_csv(test_path, parse_dates=['date'], index_col='date')
        print(f"\n测试数据: {len(test_data)} 天")
        print(f"时间范围: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"✅ 测试数据已存在: {test_path}")
    else:
        print(f"\n⚠️ 测试数据不存在: {test_path}")
        print("请先运行以下命令获取测试数据：")
        print("  python scripts/fetch_training_data.py")

    print("\n" + "=" * 70)
    print("  数据准备完成！")
    print("=" * 70)
    print(f"训练集: 2015-2019 ({len(train_data)} 天)")
    print(f"测试集: 2020-2024 (已有)")
    print("=" * 70)


if __name__ == "__main__":
    fetch_training_data()