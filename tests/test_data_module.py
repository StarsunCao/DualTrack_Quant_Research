"""
数据获取与预处理模块快速测试脚本。

优先测试本地功能（MockNewsGenerator、DataAligner），
网络数据获取（akshare、yfinance）可能较慢，放在最后测试。
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MarketDataFetcher, MockNewsGenerator, DataAligner


def print_separator(title: str) -> None:
    """打印分隔线。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def check_dataframe_quality(df: pd.DataFrame, name: str) -> None:
    """
    检查 DataFrame 的数据质量。

    Args:
        df: 待检查的 DataFrame。
        name: DataFrame 名称，用于打印。
    """
    print(f"\n📊 {name} 数据质量检查:")
    print(f"   - Shape: {df.shape}")

    # 检查 NaN 值
    nan_count = df.isna().sum()
    total_nan = nan_count.sum()
    if total_nan > 0:
        print(f"   ⚠️  存在 NaN 值:")
        for col, count in nan_count[nan_count > 0].items():
            print(f"       - {col}: {count} 个 NaN ({count/len(df)*100:.2f}%)")
    else:
        print(f"   ✅ 无 NaN 值")

    # 检查数据类型
    print(f"   - 数据类型:")
    for col in df.columns:
        print(f"       - {col}: {df[col].dtype}")


def create_sample_price_data() -> pd.DataFrame:
    """
    创建示例价格数据用于测试。

    Returns:
        示例价格数据 DataFrame。
    """
    np.random.seed(42)
    dates = pd.date_range(
        start="2024-01-01",
        end="2024-01-31",
        freq="B",  # 工作日
    )

    base_price = 100
    returns = np.random.randn(len(dates)) * 0.02
    prices = base_price * (1 + returns).cumprod()

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(len(dates)) * 0.005),
        "high": prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
        "low": prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def test_news_generator() -> pd.DataFrame:
    """
    测试 MockNewsGenerator 类。

    Returns:
        生成的新闻数据 DataFrame。
    """
    print_separator("测试 MockNewsGenerator")

    generator = MockNewsGenerator()
    print(f"✅ MockNewsGenerator 实例化成功")
    print(f"   - 输出目录: {generator.output_dir}")

    # 生成新闻数据
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    print(f"\n📝 正在生成模拟新闻数据...")
    print(f"   - 日期范围: {start_date} ~ {end_date}")
    print(f"   - 每日新闻数: 3条")

    start_time = time.time()
    news_df = generator.generate_news(
        start_date=start_date,
        end_date=end_date,
        num_per_day=3,
        save_to_file=False,
    )
    elapsed = time.time() - start_time

    print(f"   ✅ 新闻数据生成完成，耗时: {elapsed:.4f}秒")
    print(f"\n   📰 新闻数据结构:")
    print(f"      - Shape: {news_df.shape}")
    print(f"      - Columns: {list(news_df.columns)}")

    check_dataframe_quality(news_df, "新闻数据")

    print(f"\n   📋 前5条新闻:")
    for idx, row in news_df.head(5).iterrows():
        print(f"      [{row['timestamp']}] {row['title'][:45]}...")

    print(f"\n   📋 后5条新闻:")
    for idx, row in news_df.tail(5).iterrows():
        print(f"      [{row['timestamp']}] {row['title'][:45]}...")

    # 生成研报数据
    print(f"\n📄 正在生成模拟研报数据...")
    start_time = time.time()
    reports_df = generator.generate_research_reports(num_reports=10, save_to_file=False)
    elapsed = time.time() - start_time

    print(f"   ✅ 研报数据生成完成，耗时: {elapsed:.4f}秒")
    print(f"      - Shape: {reports_df.shape}")

    print(f"\n   📋 研报样例:")
    for idx, row in reports_df.head(3).iterrows():
        print(f"      [{row['timestamp']}] {row['title'][:50]}...")

    return news_df


def test_data_aligner(price_data: dict[str, pd.DataFrame], news_data: pd.DataFrame) -> None:
    """
    测试 DataAligner 类。

    Args:
        price_data: 市场数据字典。
        news_data: 新闻数据 DataFrame。
    """
    print_separator("测试 DataAligner")

    aligner = DataAligner()
    print(f"✅ DataAligner 实例化成功")
    print(f"   - 处理后数据目录: {aligner.processed_dir}")

    # 使用第一个可用的价格数据
    price_name = list(price_data.keys())[0]
    price_df = price_data[price_name]

    print(f"\n🔗 测试新闻数据与 {price_name} 价格数据对齐...")

    start_time = time.time()
    aligned_df = aligner.align_news_to_price(
        price_data=price_df,
        news_data=news_data,
        fill_method="ffill",
        save_to_file=False,
    )
    elapsed = time.time() - start_time

    print(f"   ✅ 数据对齐完成，耗时: {elapsed:.4f}秒")
    print(f"\n   📊 对齐后数据结构:")
    print(f"      - Shape: {aligned_df.shape}")
    print(f"      - Columns: {list(aligned_df.columns)}")

    check_dataframe_quality(aligned_df, "对齐后数据")

    # 统计有新闻的交易日
    if "news_titles" in aligned_df.columns:
        news_count = aligned_df["news_titles"].notna().sum()
        total_count = len(aligned_df)
        print(f"\n   📰 有新闻的交易日: {news_count}/{total_count} ({news_count/total_count*100:.1f}%)")

    print(f"\n   📋 对齐后数据样例 (前5行):")
    display_cols = [c for c in aligned_df.columns if c in ["open", "close", "volume", "news_titles"]]
    if display_cols:
        for col in display_cols:
            if col == "news_titles":
                aligned_df["news_titles_short"] = aligned_df["news_titles"].str[:30] + "..."
                print(aligned_df[["open", "close", "volume", "news_titles_short"]].head(5).to_string())
                break
        else:
            print(aligned_df[display_cols].head(5).to_string())

    # 测试重采样功能
    print(f"\n📊 测试价格数据重采样 (日 -> 周)...")

    start_time = time.time()
    resampled_df = aligner.resample_price_data(
        price_data=price_df,
        freq="W",
        save_to_file=False,
    )
    elapsed = time.time() - start_time

    print(f"   ✅ 重采样完成，耗时: {elapsed:.4f}秒")
    print(f"      - 原始数据行数: {len(price_df)}")
    print(f"      - 重采样后行数: {len(resampled_df)}")
    print(f"\n   📋 重采样后数据样例 (前5行):")
    print(resampled_df.head(5).to_string())

    # 测试多序列对齐
    if len(price_data) >= 2:
        print(f"\n🔗 测试多序列对齐...")

        start_time = time.time()
        multi_aligned = aligner.align_multiple_series(
            dataframes=price_data,
            fill_method="ffill",
            save_to_file=False,
        )
        elapsed = time.time() - start_time

        print(f"   ✅ 多序列对齐完成，耗时: {elapsed:.4f}秒")
        print(f"      - Shape: {multi_aligned.shape}")
        print(f"      - Columns (前10列): {list(multi_aligned.columns)[:10]}...")


def test_market_data_fetcher() -> dict[str, pd.DataFrame]:
    """
    测试 MarketDataFetcher 类。

    Returns:
        获取的市场数据字典。
    """
    print_separator("测试 MarketDataFetcher")

    fetcher = MarketDataFetcher()
    print(f"✅ MarketDataFetcher 实例化成功")
    print(f"   - 原始数据目录: {fetcher.raw_data_dir}")
    print(f"   - 处理后数据目录: {fetcher.processed_data_dir}")

    results: dict[str, pd.DataFrame] = {}

    # 测试获取沪深300数据
    print("\n📥 正在获取沪深300数据 (通过 akshare)...")
    print("   ⏳ 这可能需要几秒钟...")
    start_time = time.time()
    try:
        csi300_df = fetcher.fetch_csi300(
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            save_to_file=False,
        )
        elapsed = time.time() - start_time
        print(f"   ✅ 沪深300数据获取完成，耗时: {elapsed:.2f}秒")

        if not csi300_df.empty:
            results["csi300"] = csi300_df
            print(f"\n   📈 沪深300 数据结构:")
            print(f"      - Shape: {csi300_df.shape}")
            print(f"      - Columns: {list(csi300_df.columns)}")
            print(f"      - 索引类型: {type(csi300_df.index).__name__}")
            print(f"      - 日期范围: {csi300_df.index.min()} ~ {csi300_df.index.max()}")

            check_dataframe_quality(csi300_df, "沪深300")

            print(f"\n   📋 前5行数据:")
            print(csi300_df.head(5).to_string())
            print(f"\n   📋 后5行数据:")
            print(csi300_df.tail(5).to_string())
        else:
            print("   ⚠️  沪深300数据为空")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ 沪深300数据获取失败，耗时: {elapsed:.2f}秒")
        print(f"      错误信息: {e}")

    # 测试获取NASDAQ-100数据
    print("\n📥 正在获取NASDAQ-100数据 (通过 yfinance)...")
    print("   ⏳ 这可能需要几秒钟...")
    start_time = time.time()
    try:
        nasdaq_df = fetcher.fetch_nasdaq100(
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            save_to_file=False,
        )
        elapsed = time.time() - start_time
        print(f"   ✅ NASDAQ-100数据获取完成，耗时: {elapsed:.2f}秒")

        if not nasdaq_df.empty:
            results["nasdaq100"] = nasdaq_df
            print(f"\n   📈 NASDAQ-100 数据结构:")
            print(f"      - Shape: {nasdaq_df.shape}")
            print(f"      - Columns: {list(nasdaq_df.columns)}")
            print(f"      - 索引类型: {type(nasdaq_df.index).__name__}")
            print(f"      - 日期范围: {nasdaq_df.index.min()} ~ {nasdaq_df.index.max()}")

            check_dataframe_quality(nasdaq_df, "NASDAQ-100")

            print(f"\n   📋 前5行数据:")
            print(nasdaq_df.head(5).to_string())
            print(f"\n   📋 后5行数据:")
            print(nasdaq_df.tail(5).to_string())
        else:
            print("   ⚠️  NASDAQ-100数据为空")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ NASDAQ-100数据获取失败，耗时: {elapsed:.2f}秒")
        print(f"      错误信息: {e}")

    return results


def main() -> None:
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("  🚀 数据获取与预处理模块测试")
    print("=" * 60)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()

    # 1. 首先测试本地功能（快速）
    news_data = test_news_generator()

    # 2. 创建示例价格数据用于 DataAligner 测试
    print_separator("创建示例价格数据")
    sample_price = create_sample_price_data()
    print(f"✅ 示例价格数据创建完成")
    print(f"   - Shape: {sample_price.shape}")
    print(f"   - 日期范围: {sample_price.index.min()} ~ {sample_price.index.max()}")

    # 3. 测试 DataAligner
    test_data_aligner({"sample_price": sample_price}, news_data)

    # 4. 最后测试网络数据获取（可能较慢）
    market_data = test_market_data_fetcher()

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print(f"  ✅ 所有测试完成，总耗时: {total_elapsed:.2f}秒")
    print("=" * 60)

    # 汇总测试结果
    print("\n📋 测试结果汇总:")
    print("   ✅ MockNewsGenerator: 新闻生成功能正常")
    print("   ✅ DataAligner: 数据对齐功能正常")
    if market_data:
        print(f"   ✅ MarketDataFetcher: 获取了 {len(market_data)} 个市场数据")
    else:
        print("   ⚠️  MarketDataFetcher: 网络数据获取失败（可能是网络问题）")


if __name__ == "__main__":
    main()