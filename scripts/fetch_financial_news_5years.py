"""
获取5年(2020-2024)财经新闻数据 - 批量版本。

分年度获取，避免单次请求过多导致失败。
每年数据单独保存，最后合并。
"""

import pandas as pd
from pathlib import Path
import logging
from fetch_financial_news import fetch_baidu_economic_news

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def fetch_yearly_news(year: int, output_dir: Path) -> pd.DataFrame:
    """获取单年新闻数据。"""
    start_date = f"{year}0101"
    end_date = f"{year}1231"

    output_file = output_dir / f"financial_news_{year}.csv"

    if output_file.exists():
        logger.info(f"{year}年数据已存在，跳过...")
        return pd.read_csv(output_file)

    logger.info(f"\n{'='*60}")
    logger.info(f"开始获取 {year} 年财经新闻")
    logger.info(f"{'='*60}")

    df = fetch_baidu_economic_news(
        start_date=start_date,
        end_date=end_date,
        delay=1.5  # 增加延迟，避免被封
    )

    if not df.empty:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"{year}年数据已保存: {len(df)} 条")
    else:
        logger.warning(f"{year}年无数据")

    return df


def main():
    """获取2020-2024共5年的财经新闻。"""
    output_dir = Path("data/raw/financial_news")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_years_data = []

    # 逐年获取
    for year in range(2020, 2025):
        df = fetch_yearly_news(year, output_dir)
        if not df.empty:
            all_years_data.append(df)

    # 合并所有年份
    if all_years_data:
        combined = pd.concat(all_years_data, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.sort_values('date')

        # 保存合并文件
        combined_file = output_dir / "financial_news_2020_2024.csv"
        combined.to_csv(combined_file, index=False, encoding='utf-8-sig')

        logger.info(f"\n{'='*60}")
        logger.info("5年数据获取完成!")
        logger.info(f"{'='*60}")
        logger.info(f"总计: {len(combined)} 条新闻")
        logger.info(f"时间范围: {combined['date'].min()} ~ {combined['date'].max()}")
        logger.info(f"合并文件: {combined_file}")
        logger.info(f"\n按年份分布:")
        logger.info(combined.groupby(combined['date'].dt.year).size())

    else:
        logger.warning("未获取到任何数据")


if __name__ == "__main__":
    main()
