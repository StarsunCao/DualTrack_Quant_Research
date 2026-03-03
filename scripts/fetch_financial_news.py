"""
获取财经新闻数据（补充新闻联播数据）。

使用 akshare 的财经新闻接口，获取历史财经新闻数据。
数据格式与新闻联播数据保持一致，便于直接使用。

数据源：
1. 百度财经新闻 (news_economic_baidu) - 宏观经济新闻
2. 东方财富个股新闻 (stock_news_em) - 个股相关新闻

输出格式：
与新闻联播数据一致：date, title, content, source
"""

import pandas as pd
import akshare as ak
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import time
import logging

logger = logging.getLogger(__name__)


def fetch_baidu_economic_news(
    start_date: str,
    end_date: str,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    获取百度财经新闻。

    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        delay: 请求延迟（秒），避免被封

    Returns:
        新闻DataFrame，列：date, title, content, source
    """
    logger.info(f"开始获取百度财经新闻: {start_date} ~ {end_date}")

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    all_news = []
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")

        try:
            logger.info(f"正在获取 {date_str} 的新闻...")
            df = ak.news_economic_baidu(date=date_str)

            if not df.empty:
                # 标准化列名
                df = df.rename(columns={
                    'date': 'date',
                    'title': 'title',
                    'content': 'content'
                })
                df['source'] = 'baidu_economic'
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                all_news.append(df)
                logger.info(f"  获取 {len(df)} 条新闻")
            else:
                logger.warning(f"  {date_str} 无新闻数据")

            time.sleep(delay)
        except Exception as e:
            logger.error(f"  获取 {date_str} 失败: {e}")
            time.sleep(delay * 2)  # 失败后增加延迟

        current_dt += timedelta(days=1)

    if all_news:
        result = pd.concat(all_news, ignore_index=True)
        logger.info(f"百度财经新闻获取完成: 共 {len(result)} 条")
        return result
    else:
        return pd.DataFrame(columns=['date', 'title', 'content', 'source'])


def fetch_eastmoney_stock_news(
    symbols: List[str],
    max_per_symbol: int = 10
) -> pd.DataFrame:
    """
    获取东方财富个股新闻。

    Args:
        symbols: 股票代码列表（如 ['000001', '600519']）
        max_per_symbol: 每只股票最多获取的新闻数量

    Returns:
        新闻DataFrame，列：date, title, content, source
    """
    logger.info(f"开始获取东方财富个股新闻，共 {len(symbols)} 只股票")

    all_news = []

    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"[{i+1}/{len(symbols)}] 获取 {symbol} 的新闻...")
            df = ak.stock_news_em(symbol=symbol)

            if not df.empty:
                # 取前N条
                df = df.head(max_per_symbol)

                # 标准化列名
                # 东方财富新闻列名可能不同，需要适配
                if '发布时间' in df.columns:
                    df = df.rename(columns={'发布时间': 'date'})
                if '新闻标题' in df.columns:
                    df = df.rename(columns={'新闻标题': 'title'})
                if '新闻内容' in df.columns:
                    df = df.rename(columns={'新闻内容': 'content'})

                # 确保必要列存在
                if 'date' not in df.columns:
                    df['date'] = datetime.now().strftime('%Y-%m-%d')
                if 'title' not in df.columns:
                    df['title'] = ''
                if 'content' not in df.columns:
                    df['content'] = df.get('title', '')

                df['source'] = 'eastmoney_stock'
                df['symbol'] = symbol  # 标记股票代码

                all_news.append(df)
                logger.info(f"  获取 {len(df)} 条新闻")
            else:
                logger.warning(f"  {symbol} 无新闻数据")

            time.sleep(0.5)  # 避免请求过快
        except Exception as e:
            logger.error(f"  获取 {symbol} 失败: {e}")
            time.sleep(1.0)

    if all_news:
        result = pd.concat(all_news, ignore_index=True)
        logger.info(f"东方财富个股新闻获取完成: 共 {len(result)} 条")
        return result
    else:
        return pd.DataFrame(columns=['date', 'title', 'content', 'source', 'symbol'])


def fetch_financial_news_combined(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    获取综合财经新闻（百度宏观 + 东财个股）。

    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        symbols: 股票代码列表（可选，用于个股新闻）
        output_path: 输出文件路径

    Returns:
        合并后的新闻DataFrame
    """
    all_news = []

    # 1. 获取百度财经新闻（宏观经济）
    logger.info("=" * 60)
    logger.info("步骤 1: 获取百度财经新闻（宏观经济）")
    logger.info("=" * 60)
    baidu_news = fetch_baidu_economic_news(start_date, end_date)
    if not baidu_news.empty:
        all_news.append(baidu_news)

    # 2. 获取东方财富个股新闻（可选）
    if symbols:
        logger.info("\n" + "=" * 60)
        logger.info("步骤 2: 获取东方财富个股新闻")
        logger.info("=" * 60)
        eastmoney_news = fetch_eastmoney_stock_news(symbols, max_per_symbol=5)
        if not eastmoney_news.empty:
            all_news.append(eastmoney_news)

    # 合并
    if all_news:
        result = pd.concat(all_news, ignore_index=True)

        # 标准化日期格式
        result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')

        # 保存
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n新闻数据已保存至: {output_path}")

        logger.info(f"\n总计获取 {len(result)} 条新闻")
        logger.info(f"数据来源分布:\n{result['source'].value_counts()}")

        return result
    else:
        logger.warning("未获取到任何新闻数据")
        return pd.DataFrame(columns=['date', 'title', 'content', 'source'])


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 示例：获取2020年全年财经新闻
    start_date = "20200101"
    end_date = "20201231"

    # 可选：指定重点股票（沪深300前10大权重股）
    top_stocks = [
        '600519',  # 贵州茅台
        '601318',  # 中国平安
        '600036',  # 招商银行
        '601166',  # 兴业银行
        '000858',  # 五粮液
    ]

    output_path = Path("data/raw/csi300_financial_news_2020.csv")

    result = fetch_financial_news_combined(
        start_date=start_date,
        end_date=end_date,
        symbols=top_stocks,
        output_path=output_path
    )

    # 显示样例
    if not result.empty:
        print("\n新闻样例:")
        print(result.head(5).to_string())

        print("\n按日期统计:")
        print(result.groupby('date').size().describe())