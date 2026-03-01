"""
真实市场数据获取测试。

测试 akshare (A股) 和 yfinance (美股) 的实际连接性，
并保存真实数据用于后续分析。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import MarketDataFetcher


def test_csi300_real_data() -> pd.DataFrame:
    """
    测试获取沪深300指数真实数据。

    Returns:
        获取的DataFrame。
    """
    print("=" * 60)
    print("测试 1: 沪深300指数数据获取 (akshare)")
    print("=" * 60)

    fetcher = MarketDataFetcher()

    # 获取过去1年数据
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"时间范围: {start_date} ~ {end_date}")

    try:
        df = fetcher.fetch_csi300(
            start_date=start_date,
            end_date=end_date,
            save_to_file=False,  # 测试时不保存默认文件
        )

        # 数据质检
        print(f"\n数据形状: {df.shape}")
        print(f"日期范围: {df.index.min()} ~ {df.index.max()}")
        print(f"\n列信息:")
        print(df.dtypes)
        print(f"\n前3行:")
        print(df.head(3))
        print(f"\n后3行:")
        print(df.tail(3))
        print(f"\n缺失值统计:")
        print(df.isnull().sum())

        # 保存为CSV用于后续分析
        output_path = Path("data/raw/real_csi300_1y.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        print(f"\n数据已保存至: {output_path}")

        return df

    except Exception as e:
        print(f"❌ 沪深300数据获取失败: {e}")
        raise


def test_qqq_real_data() -> pd.DataFrame:
    """
    测试获取 QQQ ETF 真实数据。

    Returns:
        获取的DataFrame。
    """
    print("\n" + "=" * 60)
    print("测试 2: QQQ ETF 数据获取 (yfinance)")
    print("=" * 60)

    fetcher = MarketDataFetcher()

    # 获取过去1年数据
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"时间范围: {start_date} ~ {end_date}")

    try:
        df = fetcher.fetch_qqq(
            start_date=start_date,
            end_date=end_date,
            save_to_file=False,  # 测试时不保存默认文件
        )

        # 数据质检
        print(f"\n数据形状: {df.shape}")
        print(f"日期范围: {df.index.min()} ~ {df.index.max()}")
        print(f"\n列信息:")
        print(df.dtypes)
        print(f"\n前3行:")
        print(df.head(3))
        print(f"\n后3行:")
        print(df.tail(3))
        print(f"\n缺失值统计:")
        print(df.isnull().sum())

        # 保存为CSV用于后续分析
        output_path = Path("data/raw/real_qqq_1y.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        print(f"\n数据已保存至: {output_path}")

        return df

    except Exception as e:
        print(f"❌ QQQ 数据获取失败: {e}")
        raise


def main():
    """执行所有真实市场数据测试。"""
    print("=" * 60)
    print("真实市场数据获取测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    csi300_df = None
    qqq_df = None

    # 测试沪深300
    try:
        csi300_df = test_csi300_real_data()
        print("\n✅ 沪深300数据获取成功")
    except Exception:
        print("\n❌ 沪深300数据获取失败")

    # 测试QQQ
    try:
        qqq_df = test_qqq_real_data()
        print("\n✅ QQQ数据获取成功")
    except Exception:
        print("\n❌ QQQ数据获取失败")

    # 汇总报告
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"沪深300: {'✅ 成功' if csi300_df is not None else '❌ 失败'}")
    print(f"QQQ: {'✅ 成功' if qqq_df is not None else '❌ 失败'}")

    if csi300_df is not None:
        print(f"沪深300数据行数: {len(csi300_df)}")
    if qqq_df is not None:
        print(f"QQQ数据行数: {len(qqq_df)}")


if __name__ == "__main__":
    main()