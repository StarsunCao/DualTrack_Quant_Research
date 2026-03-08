#!/usr/bin/env python
"""
QQQ 历史数据下载脚本。

下载 QQQ 2015-2020 年 OHLCV 数据，并分割为训练集和回测集。

使用方法:
    python scripts/fetch_qqq_training_data.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_qqq_data(
    start_date: str = "2015-01-01",
    end_date: str = "2020-07-18",
    split_date: str = "2018-01-01",
    output_dir: str = "data/raw",
) -> None:
    """
    下载 QQQ 历史数据并分割为训练集和回测集。

    Args:
        start_date: 数据起始日期（训练集起始）
        end_date: 数据结束日期（回测集结束）
        split_date: 训练集/回测集分割日期
        output_dir: 输出目录
    """
    logger.info("=" * 60)
    logger.info("  QQQ 历史数据下载")
    logger.info("=" * 60)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 下载 QQQ 数据
    logger.info(f"下载 QQQ 数据: {start_date} 至 {end_date}")
    qqq = yf.Ticker("QQQ")
    df = qqq.history(start=start_date, end=end_date)

    if df.empty:
        logger.error("❌ 数据下载失败")
        return

    # 重置索引
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)

    # 确保列名统一
    df.columns = df.columns.str.lower()

    logger.info(f"✅ 数据下载成功: {len(df)} 条记录")
    logger.info(f"  时间范围: {df['date'].min()} 至 {df['date'].max()}")

    # 分割训练集和回测集
    split_dt = pd.to_datetime(split_date)

    # 处理时区问题：移除时区信息
    df["date"] = df["date"].dt.tz_localize(None)

    train_df = df[df["date"] < split_dt].copy()
    test_df = df[df["date"] >= split_dt].copy()

    # 保存训练集
    train_file = output_path / "qqq_train_2015_2017.csv"
    train_df.to_csv(train_file, index=False)
    logger.info(f"✅ 训练集已保存: {train_file}")
    logger.info(f"  时间范围: {train_df['date'].min()} 至 {train_df['date'].max()}")
    logger.info(f"  数据量: {len(train_df)} 天")

    # 保存回测集
    test_file = output_path / "qqq_test_2018_2020.csv"
    test_df.to_csv(test_file, index=False)
    logger.info(f"✅ 回测集已保存: {test_file}")
    logger.info(f"  时间范围: {test_df['date'].min()} 至 {test_df['date'].max()}")
    logger.info(f"  数据量: {len(test_df)} 天")

    # 数据验证
    logger.info("\n" + "=" * 60)
    logger.info("  数据验证")
    logger.info("=" * 60)

    # 验证训练集
    train_data = pd.read_csv(train_file, parse_dates=["date"])
    logger.info(f"训练集验证:")
    logger.info(f"  行数: {len(train_data)}")
    logger.info(f"  列数: {len(train_data.columns)}")
    logger.info(f"  列名: {list(train_data.columns)}")
    logger.info(f"  缺失值: {train_data.isnull().sum().sum()}")
    logger.info(f"  起始日期: {train_data['date'].min()}")
    logger.info(f"  结束日期: {train_data['date'].max()}")

    # 验证回测集
    test_data = pd.read_csv(test_file, parse_dates=["date"])
    logger.info(f"\n回测集验证:")
    logger.info(f"  行数: {len(test_data)}")
    logger.info(f"  列数: {len(test_data.columns)}")
    logger.info(f"  列名: {list(test_data.columns)}")
    logger.info(f"  缺失值: {test_data.isnull().sum().sum()}")
    logger.info(f"  起始日期: {test_data['date'].min()}")
    logger.info(f"  结束日期: {test_data['date'].max()}")

    # 验证与新闻数据对齐
    news_file = Path("data/raw/us_market_news/us_news_real_timestamp_2020.csv")
    if news_file.exists():
        news_data = pd.read_csv(news_file, parse_dates=["timestamp"])
        news_start = news_data["timestamp"].min()
        news_end = news_data["timestamp"].max()

        logger.info(f"\n新闻数据时间范围:")
        logger.info(f"  起始: {news_start}")
        logger.info(f"  结束: {news_end}")

        # 检查时间对齐
        test_start = test_data["date"].min()
        if news_start >= test_start:
            logger.info(f"✅ 新闻数据与回测集时间对齐正确")
        else:
            logger.warning(f"⚠️  新闻数据早于回测集起始日期")
    else:
        logger.warning(f"⚠️  新闻数据文件不存在: {news_file}")

    logger.info("\n" + "=" * 60)
    logger.info("  数据下载完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    fetch_qqq_data()