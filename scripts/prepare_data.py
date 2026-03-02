#!/usr/bin/env python
"""
数据获取与验证脚本。

自动化数据准备流程：
1. 下载 2020-2024 完整数据
2. 数据质量检查（缺失值、时间连续性）
3. 数据预处理（对齐、填充）
4. 生成数据报告

Usage:
    python scripts/prepare_data.py --symbol CSI300
    python scripts/prepare_data.py --symbol QQQ
    python scripts/prepare_data.py --all
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import MarketDataFetcher
from src.data.news_data import MockNewsGenerator
from src.data.data_aligner import DataAligner


class DataQualityChecker:
    """数据质量检查器。"""

    def __init__(self):
        self.issues = []
        self.warnings = []

    def check_missing_values(self, df: pd.DataFrame, name: str) -> dict:
        """检查缺失值。"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        result = {
            "total_rows": len(df),
            "columns_with_missing": missing[missing > 0].to_dict(),
            "missing_percentages": missing_pct[missing_pct > 0].to_dict(),
        }

        if missing.any():
            self.warnings.append(f"{name}: 存在缺失值")

        return result

    def check_time_continuity(self, df: pd.DataFrame, name: str, freq: str = "B") -> dict:
        """检查时间连续性。"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"error": "索引不是 DatetimeIndex"}

        # 生成期望的完整时间序列
        expected_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)

        # 检查缺失的日期
        missing_dates = expected_dates.difference(df.index)

        result = {
            "start_date": df.index.min().isoformat(),
            "end_date": df.index.max().isoformat(),
            "expected_records": len(expected_dates),
            "actual_records": len(df),
            "missing_records": len(missing_dates),
            "completeness_rate": round((len(df) / len(expected_dates)) * 100, 2),
        }

        if len(missing_dates) > 0:
            self.warnings.append(f"{name}: 时间序列不完整，缺失 {len(missing_dates)} 条记录")
            result["sample_missing_dates"] = [d.isoformat() for d in missing_dates[:5]]

        return result

    def check_price_reasonableness(self, df: pd.DataFrame, name: str) -> dict:
        """检查价格合理性。"""
        issues = []

        # 检查 OHLC 关系
        if "high" in df.columns and "low" in df.columns:
            invalid_hl = df[df["high"] < df["low"]]
            if len(invalid_hl) > 0:
                issues.append(f"{len(invalid_hl)} 条记录 high < low")

        if "close" in df.columns and "high" in df.columns:
            invalid_close_high = df[df["close"] > df["high"]]
            if len(invalid_close_high) > 0:
                issues.append(f"{len(invalid_close_high)} 条记录 close > high")

        if "close" in df.columns and "low" in df.columns:
            invalid_close_low = df[df["close"] < df["low"]]
            if len(invalid_close_low) > 0:
                issues.append(f"{len(invalid_close_low)} 条记录 close < low")

        # 检查价格范围
        if "close" in df.columns:
            price_stats = df["close"].describe()
            min_price = price_stats["min"]
            max_price = price_stats["max"]

            if min_price <= 0:
                issues.append(f"存在非正价格: {min_price}")

            # 检查极端价格变动
            returns = df["close"].pct_change().abs()
            extreme_moves = returns[returns > 0.2]  # 单日变动超过 20%
            if len(extreme_moves) > 0:
                self.warnings.append(f"{name}: 存在 {len(extreme_moves)} 条极端价格变动记录 (>20%)")

        result = {
            "price_range": {"min": float(min_price), "max": float(max_price)} if "close" in df.columns else None,
            "issues": issues,
            "valid": len(issues) == 0,
        }

        if issues:
            self.issues.extend([f"{name}: {issue}" for issue in issues])

        return result

    def check_volume(self, df: pd.DataFrame, name: str) -> dict:
        """检查成交量。"""
        if "volume" not in df.columns:
            return {"error": "无成交量数据"}

        volume_stats = df["volume"].describe()

        # 检查零成交量
        zero_volume = (df["volume"] == 0).sum()

        result = {
            "total_records": len(df),
            "zero_volume_records": int(zero_volume),
            "zero_volume_percentage": round((zero_volume / len(df)) * 100, 2),
            "volume_stats": {
                "mean": float(volume_stats["mean"]),
                "std": float(volume_stats["std"]),
                "min": float(volume_stats["min"]),
                "max": float(volume_stats["max"]),
            },
        }

        if zero_volume > 0:
            self.warnings.append(f"{name}: 存在 {zero_volume} 条零成交量记录")

        return result

    def generate_report(self) -> dict:
        """生成质量检查报告。"""
        return {
            "timestamp": datetime.now().isoformat(),
            "issues": self.issues,
            "warnings": self.warnings,
            "has_critical_issues": len(self.issues) > 0,
            "has_warnings": len(self.warnings) > 0,
        }


def prepare_market_data(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
) -> Optional[Path]:
    """
    准备市场数据。

    Args:
        symbol: 交易标的 (CSI300/QQQ)
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录

    Returns:
        保存的文件路径，失败返回 None
    """
    logger.info(f"准备市场数据: {symbol} ({start_date} ~ {end_date})")

    fetcher = MarketDataFetcher()

    try:
        if symbol == "CSI300":
            df = fetcher.fetch_csi300(start_date=start_date, end_date=end_date)
        elif symbol == "QQQ":
            df = fetcher.fetch_qqq(start_date=start_date, end_date=end_date)
        else:
            logger.error(f"不支持的标的: {symbol}")
            return None

        if df.empty:
            logger.error("获取的数据为空")
            return None

        # 质量检查
        logger.info("执行数据质量检查...")
        checker = DataQualityChecker()

        missing_check = checker.check_missing_values(df, symbol)
        continuity_check = checker.check_time_continuity(df, symbol)
        price_check = checker.check_price_reasonableness(df, symbol)
        volume_check = checker.check_volume(df, symbol)

        quality_report = checker.generate_report()

        if quality_report["has_critical_issues"]:
            logger.error("数据存在严重问题:")
            for issue in quality_report["issues"]:
                logger.error(f"  - {issue}")
            return None

        if quality_report["has_warnings"]:
            logger.warning("数据存在警告:")
            for warning in quality_report["warnings"]:
                logger.warning(f"  - {warning}")

        # 保存数据
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"{symbol.lower()}_ohlcv.csv"
        df.to_csv(file_path)

        logger.info(f"市场数据已保存: {file_path}")
        logger.info(f"  记录数: {len(df)}")
        logger.info(f"  日期范围: {df.index.min()} ~ {df.index.max()}")
        logger.info(f"  完整率: {continuity_check['completeness_rate']}%")

        return file_path

    except Exception as e:
        logger.error(f"准备市场数据失败: {e}")
        return None


def prepare_news_data(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
    use_mock: bool = True,
) -> Optional[Path]:
    """
    准备新闻数据。

    Args:
        symbol: 交易标的
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        use_mock: 是否使用模拟数据

    Returns:
        保存的文件路径，失败返回 None
    """
    logger.info(f"准备新闻数据: {symbol} ({start_date} ~ {end_date})")

    if use_mock:
        logger.info("使用模拟新闻数据生成器")
        generator = MockNewsGenerator()
        news_df = generator.generate_mock_news(
            start_date=start_date,
            end_date=end_date,
            symbols=[symbol],
            avg_news_per_day=3,
        )
    else:
        # TODO: 实现真实新闻数据获取
        logger.warning("真实新闻数据获取尚未实现，使用模拟数据")
        generator = MockNewsGenerator()
        news_df = generator.generate_mock_news(
            start_date=start_date,
            end_date=end_date,
            symbols=[symbol],
        )

    if news_df.empty:
        logger.error("生成的新闻数据为空")
        return None

    # 保存数据
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{symbol.lower()}_news.csv"
    news_df.to_csv(file_path, index=False)

    logger.info(f"新闻数据已保存: {file_path}")
    logger.info(f"  记录数: {len(news_df)}")

    return file_path


def generate_data_report(
    symbol: str,
    market_data_path: Optional[Path],
    news_data_path: Optional[Path],
    output_dir: str = "docs/output",
) -> Path:
    """
    生成数据报告。

    Args:
        symbol: 交易标的
        market_data_path: 市场数据文件路径
        news_data_path: 新闻数据文件路径
        output_dir: 输出目录

    Returns:
        报告文件路径
    """
    logger.info("生成数据报告...")

    report = {
        "report_type": "Data Preparation Report",
        "symbol": symbol,
        "generated_at": datetime.now().isoformat(),
        "market_data": None,
        "news_data": None,
    }

    if market_data_path and market_data_path.exists():
        df = pd.read_csv(market_data_path, parse_dates=["date"], index_col="date")

        # 计算基本统计
        returns = df["close"].pct_change().dropna()

        report["market_data"] = {
            "file_path": str(market_data_path),
            "file_size_mb": round(market_data_path.stat().st_size / (1024 * 1024), 2),
            "record_count": len(df),
            "date_range": {
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat(),
            },
            "price_statistics": {
                "open_mean": round(df["open"].mean(), 2),
                "high_mean": round(df["high"].mean(), 2),
                "low_mean": round(df["low"].mean(), 2),
                "close_mean": round(df["close"].mean(), 2),
                "close_min": round(df["close"].min(), 2),
                "close_max": round(df["close"].max(), 2),
            },
            "volume_statistics": {
                "mean": round(df["volume"].mean(), 0),
                "median": round(df["volume"].median(), 0),
                "max": int(df["volume"].max()),
            },
            "return_statistics": {
                "mean": round(returns.mean() * 100, 4),
                "std": round(returns.std() * 100, 4),
                "min": round(returns.min() * 100, 4),
                "max": round(returns.max() * 100, 4),
                "annualized_return": round(returns.mean() * 252 * 100, 2),
                "annualized_volatility": round(returns.std() * np.sqrt(252) * 100, 2),
            },
        }

    if news_data_path and news_data_path.exists():
        df = pd.read_csv(news_data_path, parse_dates=["timestamp"])

        report["news_data"] = {
            "file_path": str(news_data_path),
            "file_size_mb": round(news_data_path.stat().st_size / (1024 * 1024), 2),
            "record_count": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            },
            "sentiment_distribution": df["sentiment_label"].value_counts().to_dict() if "sentiment_label" in df.columns else None,
        }

    # 保存报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / f"data_report_{symbol.lower()}.json"
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"数据报告已保存: {report_path}")

    # 同时打印报告摘要
    print("\n" + "=" * 60)
    print("数据准备报告摘要")
    print("=" * 60)
    print(f"标的: {symbol}")
    print(f"生成时间: {report['generated_at']}")

    if report["market_data"]:
        md = report["market_data"]
        print(f"\n市场数据:")
        print(f"  记录数: {md['record_count']}")
        print(f"  日期范围: {md['date_range']['start'][:10]} ~ {md['date_range']['end'][:10]}")
        print(f"  价格范围: {md['price_statistics']['close_min']} ~ {md['price_statistics']['close_max']}")
        print(f"  年化收益: {md['return_statistics']['annualized_return']}%")
        print(f"  年化波动: {md['return_statistics']['annualized_volatility']}%")

    if report["news_data"]:
        nd = report["news_data"]
        print(f"\n新闻数据:")
        print(f"  记录数: {nd['record_count']}")
        if nd["sentiment_distribution"]:
            print(f"  情感分布: {nd['sentiment_distribution']}")

    print("=" * 60)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="DualTrack 数据准备脚本")
    parser.add_argument("--symbol", "-s", default="CSI300", help="交易标的 (CSI300/QQQ)")
    parser.add_argument("--start", default="2020-01-01", help="开始日期")
    parser.add_argument("--end", default="2024-12-31", help="结束日期")
    parser.add_argument("--all", action="store_true", help="准备所有标的的数据")
    parser.add_argument("--output-dir", default="data/raw", help="输出目录")
    parser.add_argument("--skip-market", action="store_true", help="跳过市场数据")
    parser.add_argument("--skip-news", action="store_true", help="跳过新闻数据")

    args = parser.parse_args()

    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    if args.all:
        symbols = ["CSI300", "QQQ"]
    else:
        symbols = [args.symbol]

    for symbol in symbols:
        logger.info(f"{'='*60}")
        logger.info(f"开始准备 {symbol} 的数据")
        logger.info(f"{'='*60}")

        market_path = None
        news_path = None

        # 准备市场数据
        if not args.skip_market:
            market_path = prepare_market_data(
                symbol=symbol,
                start_date=args.start,
                end_date=args.end,
                output_dir=args.output_dir,
            )

        # 准备新闻数据
        if not args.skip_news:
            news_path = prepare_news_data(
                symbol=symbol,
                start_date=args.start,
                end_date=args.end,
                output_dir=args.output_dir,
            )

        # 生成报告
        if market_path or news_path:
            generate_data_report(
                symbol=symbol,
                market_data_path=market_path,
                news_data_path=news_path,
            )

    logger.info("数据准备完成！")


if __name__ == "__main__":
    main()
