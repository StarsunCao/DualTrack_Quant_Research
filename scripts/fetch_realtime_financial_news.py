"""
获取实时财经新闻（补充现有历史数据）。

使用财联社、同花顺等接口获取最新新闻，无需Cookie。
注意：这些接口只能获取最近的新闻，无法获取历史数据。
"""

import pandas as pd
import akshare as ak
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_cls_news(max_items: int = 100) -> pd.DataFrame:
    """获取财联社新闻。"""
    logger.info("获取财联社新闻...")
    try:
        df = ak.stock_info_global_cls()
        df = df.head(max_items)

        # 标准化列名
        result = pd.DataFrame({
            'date': df['发布日期'],
            'title': df['标题'],
            'content': df['内容'],
            'source': 'cls',  # 财联社
            'time': df['发布时间']
        })

        logger.info(f"获取 {len(result)} 条财联社新闻")
        return result
    except Exception as e:
        logger.error(f"财联社新闻获取失败: {e}")
        return pd.DataFrame()


def fetch_ths_news(max_items: int = 100) -> pd.DataFrame:
    """获取同花顺新闻。"""
    logger.info("获取同花顺新闻...")
    try:
        df = ak.stock_info_global_ths()
        df = df.head(max_items)

        # 标准化列名
        result = pd.DataFrame({
            'date': datetime.now().strftime('%Y-%m-%d'),  # 同花顺只有发布时间
            'title': df['标题'],
            'content': df['内容'],
            'source': 'ths',  # 同花顺
            'time': df['发布时间']
        })

        logger.info(f"获取 {len(result)} 条同花顺新闻")
        return result
    except Exception as e:
        logger.error(f"同花顺新闻获取失败: {e}")
        return pd.DataFrame()


def main():
    """获取实时财经新闻并保存。"""
    output_dir = Path("data/raw/realtime_news")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_news = []

    # 获取各源新闻
    cls_df = fetch_cls_news(max_items=100)
    if not cls_df.empty:
        all_news.append(cls_df)

    ths_df = fetch_ths_news(max_items=100)
    if not ths_df.empty:
        all_news.append(ths_df)

    if all_news:
        combined = pd.concat(all_news, ignore_index=True)

        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"realtime_news_{timestamp}.csv"
        combined.to_csv(output_file, index=False, encoding='utf-8-sig')

        logger.info(f"\n实时新闻获取完成!")
        logger.info(f"总计: {len(combined)} 条")
        logger.info(f"保存至: {output_file}")
        logger.info(f"\n数据来源分布:")
        logger.info(combined['source'].value_counts())
    else:
        logger.warning("未获取到任何新闻")


if __name__ == "__main__":
    main()
