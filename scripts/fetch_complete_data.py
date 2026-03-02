#!/usr/bin/env python
"""
完整数据获取脚本（多线程加速）。

自动获取所有可纳入回测框架的数据：
- OHLCV价格数据
- CCTV新闻联播
- A股公告数据
- 宏观指标（CPI、PMI、利率）
- 北向资金流向

统一格式，支持断点续传。
"""

import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import warnings

import pandas as pd
import numpy as np
import akshare as ak
import requests
from bs4 import BeautifulSoup

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 数据时间范围
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# 输出目录
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv_data(symbol: str = "sh000300") -> pd.DataFrame:
    """
    获取OHLCV数据。

    格式统一：
    - Index: date (DatetimeIndex)
    - Columns: open, high, low, close, volume, symbol
    """
    print("\n[1/5] 获取OHLCV价格数据...")

    # 检查缓存（使用CSV格式与回测框架兼容）
    cache_file = RAW_DIR / "real_csi300_5y.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["date"])
        df = df.set_index("date")
        print(f"  ✓ 从缓存加载: {len(df)} 条")
        return df

    # 获取数据
    df = ak.stock_zh_index_daily(symbol=symbol)

    # 标准化格式
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["symbol"] = "CSI300"

    # 日期过滤
    df = df.loc[START_DATE:END_DATE]

    # 保存为CSV格式（与回测框架兼容）
    cache_file = RAW_DIR / "real_csi300_5y.csv"
    df.to_csv(cache_file)
    print(f"  ✓ 获取完成: {len(df)} 条")
    print(f"  ✓ 保存至: {cache_file}")

    return df


def fetch_cctv_news_month(year: int, month: int) -> pd.DataFrame:
    """获取单月CCTV新闻联播（用于多线程）。"""
    chunk_file = RAW_DIR / "news_chunks" / f"cctv_{year}_{month:02d}.csv"
    chunk_file.parent.mkdir(exist_ok=True)

    if chunk_file.exists():
        return pd.read_csv(chunk_file, parse_dates=["timestamp"])

    # 计算当月日期范围
    start_dt = datetime(year, month, 1)
    if month == 12:
        end_dt = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_dt = datetime(year, month + 1, 1) - timedelta(days=1)

    all_news = []
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        try:
            df = ak.news_cctv(date=date_str)
            if not df.empty:
                df = df.rename(columns={"date": "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d")
                all_news.append(df)
        except Exception:
            pass
        current_dt += timedelta(days=1)
        time.sleep(0.5)  # 避免限流

    if all_news:
        result = pd.concat(all_news, ignore_index=True)
        result.to_csv(chunk_file, index=False)
        return result
    else:
        return pd.DataFrame(columns=["timestamp", "title", "content"])


def fetch_news_data_multithreaded() -> pd.DataFrame:
    """
    获取CCTV新闻联播（多线程加速）。

    格式统一：
    - Columns: timestamp, title, content
    """
    print("\n[2/5] 获取CCTV新闻联播（多线程）...")

    # 检查完整缓存
    cache_file = RAW_DIR / "csi300_news_cctv_2020_2024.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        print(f"  ✓ 从缓存加载: {len(df)} 条")
        return df

    # 生成月份列表
    months = []
    for year in range(2020, 2025):
        for month in range(1, 13):
            months.append((year, month))

    # 多线程获取
    all_news = []
    max_workers = 4  # 控制并发，避免被封

    print(f"  启动 {max_workers} 线程获取 {len(months)} 个月份数据...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_cctv_news_month, y, m): (y, m)
                   for y, m in months}

        for future in as_completed(futures):
            year, month = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    all_news.append(df)
                    print(f"  ✓ {year}-{month:02d}: {len(df)} 条")
            except Exception as e:
                print(f"  ✗ {year}-{month:02d}: 失败 - {e}")

    if all_news:
        result = pd.concat(all_news, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)

        # 清洗数据
        result = result.drop_duplicates(subset=["timestamp", "title"])
        result = result[result["title"].str.len() >= 5]

        result.to_csv(cache_file, index=False)
        print(f"  ✓ 合并完成: {len(result)} 条")
        print(f"  ✓ 保存至: {cache_file}")
        return result
    else:
        return pd.DataFrame(columns=["timestamp", "title", "content"])


def fetch_stock_notices_multithreaded() -> pd.DataFrame:
    """
    获取A股公告数据（多线程）。

    格式统一：
    - Columns: timestamp, title, content, type
    """
    print("\n[3/5] 获取A股公告数据（多线程）...")

    cache_file = RAW_DIR / "csi300_notices_2020_2024.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        print(f"  ✓ 从缓存加载: {len(df)} 条")
        return df

    notice_types = ["重大事项", "风险提示"]
    all_notices = []

    # 按季度分块获取
    quarters = []
    for year in range(2020, 2025):
        for q in range(1, 5):
            quarters.append((year, q))

    def fetch_quarter(year: int, quarter: int) -> pd.DataFrame:
        """获取单季度公告（提取关键信息）。"""
        import requests
        from bs4 import BeautifulSoup
        import re

        chunk_file = RAW_DIR / "notice_chunks" / f"notices_{year}_Q{quarter}.csv"
        chunk_file.parent.mkdir(exist_ok=True)

        if chunk_file.exists():
            return pd.read_csv(chunk_file, parse_dates=["timestamp"])

        # 计算季度日期范围
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        start_dt = datetime(year, start_month, 1)
        if quarter == 4:
            end_dt = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_dt = datetime(year, end_month + 1, 1) - timedelta(days=1)

        quarter_notices = []
        current_dt = start_dt

        def extract_key_info(url: str, title: str) -> str:
            """提取公告关键信息。"""
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.encoding = 'utf-8'

                soup = BeautifulSoup(response.text, 'html.parser')

                # 获取所有文本
                text = soup.get_text(separator='\n')

                # 提取关键信息的模式
                key_info = []

                # 1. 公告类型（从标题提取）
                if '解除质押' in title:
                    key_info.append('股份解除质押')
                elif '质押' in title:
                    key_info.append('股份质押')
                elif '重组' in title or '并购' in title:
                    key_info.append('重大资产重组')
                elif '中标' in title:
                    key_info.append('项目中标')
                elif '投资' in title:
                    key_info.append('对外投资')

                # 2. 提取金额/比例信息
                patterns = [
                    r'(\d+(?:\.\d+)?万?股)',  # 股数
                    r'(\d+(?:\.\d+)?%)',  # 百分比
                    r'(\d+(?:\.\d+)?万?元)',  # 金额
                    r'(\d+(?:\.\d+)?亿)',  # 亿元
                ]

                amounts = []
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    amounts.extend(matches[:3])  # 每种类型最多3个

                if amounts:
                    key_info.append('、'.join(amounts[:5]))  # 总共最多5个数据

                # 3. 风险评估关键词
                if '不存在平仓风险' in text or '无平仓风险' in text:
                    key_info.append('无平仓风险')
                elif '存在平仓风险' in text:
                    key_info.append('存在平仓风险')

                # 4. 提取关键主体（前100行）
                lines = [line.strip() for line in text.split('\n') if line.strip()][:50]
                for line in lines:
                    if '关于' in line and '的公告' in line and len(line) < 50:
                        # 提取公告核心主题
                        core = line.replace('关于', '').replace('的公告', '').strip()
                        if core and len(core) < 30:
                            key_info.insert(0, core)
                            break

                return ' | '.join(key_info) if key_info else title

            except Exception as e:
                logger.warning(f"抓取公告关键信息失败: {url}, 错误: {e}")
                return title

        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            for notice_type in notice_types:
                try:
                    df = ak.stock_notice_report(symbol=notice_type, date=date_str)
                    if not df.empty:
                        # 准备数据
                        records = []
                        for idx, row in df.iterrows():
                            # 提取关键信息
                            url = row.get('网址', '')
                            title = row["公告标题"]

                            # 抓取关键信息（如果失败则使用标题）
                            content_text = extract_key_info(url, title) if url else title

                            records.append({
                                "timestamp": pd.to_datetime(row["公告日期"]),
                                "title": title,
                                "content": content_text,
                                "type": notice_type,
                            })

                        if records:
                            std_df = pd.DataFrame(records)
                            quarter_notices.append(std_df)

                except Exception as e:
                    logger.warning(f"获取公告失败 {date_str} {notice_type}: {e}")

            current_dt += timedelta(days=1)
            time.sleep(0.5)  # 避免请求过快

        if quarter_notices:
            result = pd.concat(quarter_notices, ignore_index=True)
            result.to_csv(chunk_file, index=False)
            return result
        return pd.DataFrame(columns=["timestamp", "title", "content", "type"])

    print(f"  启动多线程获取 {len(quarters)} 个季度数据...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_quarter, y, q): (y, q)
                   for y, q in quarters}

        for future in as_completed(futures):
            year, quarter = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    all_notices.append(df)
                    print(f"  ✓ {year}Q{quarter}: {len(df)} 条")
            except Exception as e:
                print(f"  ✗ {year}Q{quarter}: 失败 - {e}")

    if all_notices:
        result = pd.concat(all_notices, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)
        result = result.drop_duplicates(subset=["timestamp", "title"])

        result.to_csv(cache_file, index=False)
        print(f"  ✓ 合并完成: {len(result)} 条")
        print(f"  ✓ 保存至: {cache_file}")
        return result
    else:
        return pd.DataFrame(columns=["timestamp", "title", "content", "type"])


def fetch_macro_data() -> dict[str, pd.DataFrame]:
    """
    获取宏观经济数据。

    返回字典，包含：
    - cpi: 居民消费价格指数
    - pmi: 制造业采购经理指数
    - rates: LPR利率
    """
    print("\n[4/5] 获取宏观经济数据...")

    results = {}

    # CPI
    try:
        cache_file = RAW_DIR / "macro_cpi_2020_2024.csv"
        if cache_file.exists():
            cpi = pd.read_csv(cache_file, parse_dates=["timestamp"])
        else:
            cpi = ak.macro_china_cpi()
            # 实际列: ['月份', '全国-当月', '全国-同比增长', '全国-环比增长', ...]
            cpi = cpi[["月份", "全国-同比增长", "全国-环比增长"]].copy()
            cpi.columns = ["timestamp", "cpi_yoy", "cpi_mom"]
            # 解析中文日期格式 "2026年01月份"
            cpi["timestamp"] = pd.to_datetime(cpi["timestamp"].str.replace("年", "-").str.replace("月份", "-01"), errors="coerce")
            cpi = cpi.dropna(subset=["timestamp"])
            cpi = cpi[(cpi["timestamp"] >= START_DATE) & (cpi["timestamp"] <= END_DATE)]
            cpi.to_csv(cache_file, index=False)
        results["cpi"] = cpi
        print(f"  ✓ CPI: {len(cpi)} 条")
    except Exception as e:
        print(f"  ✗ CPI: {e}")

    # PMI
    try:
        cache_file = RAW_DIR / "macro_pmi_2020_2024.csv"
        if cache_file.exists():
            pmi = pd.read_csv(cache_file, parse_dates=["timestamp"])
        else:
            pmi = ak.macro_china_pmi()
            # 实际列: ['月份', '制造业-指数', '制造业-同比增长', ...]
            pmi = pmi[["月份", "制造业-指数", "制造业-同比增长"]].copy()
            pmi.columns = ["timestamp", "pmi", "pmi_change"]
            # 解析中文日期格式
            pmi["timestamp"] = pd.to_datetime(pmi["timestamp"].str.replace("年", "-").str.replace("月份", "-01"), errors="coerce")
            pmi = pmi.dropna(subset=["timestamp"])
            pmi = pmi[(pmi["timestamp"] >= START_DATE) & (pmi["timestamp"] <= END_DATE)]
            pmi.to_csv(cache_file, index=False)
        results["pmi"] = pmi
        print(f"  ✓ PMI: {len(pmi)} 条")
    except Exception as e:
        print(f"  ✗ PMI: {e}")

    # LPR利率
    try:
        cache_file = RAW_DIR / "macro_lpr_2020_2024.csv"
        if cache_file.exists():
            lpr = pd.read_csv(cache_file, parse_dates=["timestamp"])
        else:
            lpr = ak.macro_china_lpr()
            # 实际列: ['TRADE_DATE', 'LPR1Y', 'LPR5Y', 'RATE_1', 'RATE_2']
            lpr = lpr[["TRADE_DATE", "LPR1Y", "LPR5Y"]].copy()
            lpr.columns = ["timestamp", "lpr_1y", "lpr_5y"]
            lpr["timestamp"] = pd.to_datetime(lpr["timestamp"], errors="coerce")
            lpr = lpr.dropna(subset=["timestamp"])
            lpr = lpr[(lpr["timestamp"] >= START_DATE) & (lpr["timestamp"] <= END_DATE)]
            lpr.to_csv(cache_file, index=False)
        results["lpr"] = lpr
        print(f"  ✓ LPR: {len(lpr)} 条")
    except Exception as e:
        print(f"  ✗ LPR: {e}")

    return results


def fetch_northbound_flow() -> pd.DataFrame:
    """
    获取北向资金流向数据。

    格式：
    - timestamp: 日期
    - net_flow: 净流入金额（亿元）
    - cumulative: 累计净流入
    """
    print("\n[5/5] 获取北向资金流向...")

    cache_file = RAW_DIR / "northbound_flow_2020_2024.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        print(f"  ✓ 从缓存加载: {len(df)} 条")
        return df

    try:
        # 获取北向资金历史数据
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        # 实际列: ['日期', '当日成交净买额', '买入成交额', '卖出成交额', '历史累计净买额', ...]
        df = df[["日期", "当日成交净买额", "历史累计净买额"]].copy()
        df.columns = ["timestamp", "net_flow", "cumulative"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= START_DATE) & (df["timestamp"] <= END_DATE)]

        # 标准化格式（已经是亿元）
        df["net_flow"] = pd.to_numeric(df["net_flow"], errors="coerce")
        df["cumulative"] = pd.to_numeric(df["cumulative"], errors="coerce")
        df = df.dropna()

        df.to_csv(cache_file, index=False)
        print(f"  ✓ 获取完成: {len(df)} 条")
        print(f"  ✓ 保存至: {cache_file}")
        return df
    except Exception as e:
        print(f"  ✗ 获取失败: {e}")
        return pd.DataFrame(columns=["timestamp", "net_flow", "cumulative"])


def macro_to_news_format(macro_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    将宏观数据转换为新闻格式以便聚合。

    Args:
        macro_data: 包含cpi/pmi/lpr的字典

    Returns:
        新闻格式的DataFrame，source标记为'macro'
    """
    news_items = []

    # CPI 转为新闻
    if "cpi" in macro_data and not macro_data["cpi"].empty:
        cpi = macro_data["cpi"].copy()
        for _, row in cpi.iterrows():
            title = f"CPI同比{row['cpi_yoy']:.1f}%"
            content = f"CPI同比{row['cpi_yoy']:.1f}%，环比{row['cpi_mom']:.1f}%。居民消费价格指数反映通胀水平。"
            news_items.append({
                "timestamp": row["timestamp"],
                "title": title,
                "content": content,
                "source": "macro"
            })

    # PMI 转为新闻
    if "pmi" in macro_data and not macro_data["pmi"].empty:
        pmi = macro_data["pmi"].copy()
        for _, row in pmi.iterrows():
            title = f"PMI{row['pmi']:.1f}"
            change = row.get('pmi_change', 0)
            content = f"制造业PMI{row['pmi']:.1f}，同比变化{change:.1f}%。PMI>50表示扩张。"
            news_items.append({
                "timestamp": row["timestamp"],
                "title": title,
                "content": content,
                "source": "macro"
            })

    # LPR 转为新闻
    if "lpr" in macro_data and not macro_data["lpr"].empty:
        lpr = macro_data["lpr"].copy()
        for _, row in lpr.iterrows():
            title = f"LPR_{row['lpr_1y']:.2f}%"
            content = f"1年期LPR{row['lpr_1y']:.2f}%，5年期{row['lpr_5y']:.2f}%。贷款市场报价利率影响融资成本。"
            news_items.append({
                "timestamp": row["timestamp"],
                "title": title,
                "content": content,
                "source": "macro"
            })

    return pd.DataFrame(news_items) if news_items else pd.DataFrame(columns=["timestamp", "title", "content", "source"])


def northbound_to_news_format(northbound_df: pd.DataFrame) -> pd.DataFrame:
    """
    将北向资金数据转换为新闻格式。

    Args:
        northbound_df: 北向资金DataFrame

    Returns:
        新闻格式的DataFrame，source标记为'northbound'
    """
    if northbound_df.empty:
        return pd.DataFrame(columns=["timestamp", "title", "content", "source"])

    news_items = []
    for _, row in northbound_df.iterrows():
        flow = row["net_flow"]
        direction = "净流入" if flow > 0 else "净流出"
        title = f"北向{direction}{abs(flow):.1f}亿"
        content = f"北向资金当日{direction}{abs(flow):.1f}亿元，累计净流入{row['cumulative']:.1f}亿元。反映外资对A股态度。"
        news_items.append({
            "timestamp": row["timestamp"],
            "title": title,
            "content": content,
            "source": "northbound"
        })

    return pd.DataFrame(news_items)


def create_combined_news(
    news_df: pd.DataFrame,
    notices_df: pd.DataFrame,
    macro_news_df: pd.DataFrame,
    northbound_news_df: pd.DataFrame
) -> pd.DataFrame:
    """
    合并所有信息源：新闻、公告、宏观数据、北向资金。

    输出格式：
    - timestamp: 时间戳
    - title: 标题
    - content: 内容
    - source: 来源（cctv/notice/macro/northbound）
    """
    print("\n合并所有信息源...")

    # 标准化新闻数据
    news_df = news_df.copy()
    news_df["source"] = "cctv"

    # 标准化公告数据
    notices_df = notices_df.copy()
    notices_df["source"] = "notice"
    if "type" in notices_df.columns:
        notices_df["content"] = notices_df["type"] + " | " + notices_df["content"]
        notices_df = notices_df.drop(columns=["type"])

    # 合并所有数据源
    all_data = [news_df, notices_df]
    if not macro_news_df.empty:
        all_data.append(macro_news_df)
    if not northbound_news_df.empty:
        all_data.append(northbound_news_df)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # 去重（同一天同一标题）
    combined = combined.drop_duplicates(subset=["timestamp", "title"], keep="first")

    # 保存
    output_file = RAW_DIR / "csi300_news_combined_2020_2024.csv"
    combined.to_csv(output_file, index=False)

    print(f"  ✓ 合并完成: {len(combined)} 条")
    print(f"    - CCTV新闻: {len(news_df)} 条")
    print(f"    - 公告: {len(notices_df)} 条")
    if not macro_news_df.empty:
        print(f"    - 宏观数据: {len(macro_news_df)} 条")
    if not northbound_news_df.empty:
        print(f"    - 北向资金: {len(northbound_news_df)} 条")
    print(f"  ✓ 保存至: {output_file}")

    return combined


def validate_all_data(ohlcv: pd.DataFrame, news: pd.DataFrame, macro: dict) -> bool:
    """验证所有数据格式正确。"""
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)

    # 验证OHLCV
    print("\n1. OHLCV验证:")
    assert "open" in ohlcv.columns, "缺少open列"
    assert "high" in ohlcv.columns, "缺少high列"
    assert "low" in ohlcv.columns, "缺少low列"
    assert "close" in ohlcv.columns, "缺少close列"
    assert "volume" in ohlcv.columns, "缺少volume列"
    assert "symbol" in ohlcv.columns, "缺少symbol列"
    print(f"  ✓ 格式正确: {len(ohlcv)} 条")

    # 验证新闻
    print("\n2. 新闻数据验证:")
    assert "timestamp" in news.columns, "缺少timestamp列"
    assert "title" in news.columns, "缺少title列"
    assert "content" in news.columns, "缺少content列"
    assert "source" in news.columns, "缺少source列"
    print(f"  ✓ 格式正确: {len(news)} 条")

    # 验证宏观数据
    print("\n3. 宏观数据验证:")
    for name, df in macro.items():
        if not df.empty:
            assert "timestamp" in df.columns, f"{name}缺少timestamp列"
            print(f"  ✓ {name}: {len(df)} 条")

    print("\n✓ 所有数据格式验证通过！")
    return True


def load_existing_news() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载已存在的新闻和公告数据。"""
    news_file = RAW_DIR / "csi300_news_cctv_2020_2024.csv"
    notices_file = RAW_DIR / "csi300_notices_2020_2024.csv"

    news = pd.DataFrame(columns=["timestamp", "title", "content"])
    notices = pd.DataFrame(columns=["timestamp", "title", "content", "type"])

    if news_file.exists():
        news = pd.read_csv(news_file, parse_dates=["timestamp"])
        print(f"  ✓ 加载已有CCTV新闻: {len(news)} 条")
    else:
        print(f"  ✗ CCTV新闻不存在")

    if notices_file.exists():
        notices = pd.read_csv(notices_file, parse_dates=["timestamp"])
        print(f"  ✓ 加载已有公告: {len(notices)} 条")
    else:
        print(f"  ✗ 公告数据不存在")

    return news, notices


def main():
    """执行完整的数据获取流程。"""
    print("=" * 60)
    print("DualTrack - 完整数据获取（多线程）")
    print("=" * 60)
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        # 1. 获取OHLCV（如果存在则使用缓存）
        ohlcv = fetch_ohlcv_data()

        # 检查已有数据
        news_file = RAW_DIR / "csi300_news_cctv_2020_2024.csv"
        notices_file = RAW_DIR / "csi300_notices_2020_2024.csv"
        macro_cpi_file = RAW_DIR / "macro_cpi_2020_2024.csv"
        macro_pmi_file = RAW_DIR / "macro_pmi_2020_2024.csv"
        macro_lpr_file = RAW_DIR / "macro_lpr_2020_2024.csv"
        northbound_file = RAW_DIR / "northbound_flow_2020_2024.csv"

        has_news = news_file.exists()
        has_notices = notices_file.exists()
        has_macro = macro_cpi_file.exists() and macro_pmi_file.exists() and macro_lpr_file.exists()
        has_northbound = northbound_file.exists()

        print("\n数据状态检查:")
        print(f"  CCTV新闻: {'✓' if has_news else '✗'}")
        print(f"  公告: {'✓' if has_notices else '✗'}")
        print(f"  宏观数据: {'✓' if has_macro else '✗'}")
        print(f"  北向资金: {'✓' if has_northbound else '✗'}")

        # 2. 获取或加载CCTV新闻
        if has_news:
            news = pd.read_csv(news_file, parse_dates=["timestamp"])
            print(f"\n[2/5] 使用已有CCTV新闻: {len(news)} 条")
        else:
            news = fetch_news_data_multithreaded()

        # 3. 获取或加载公告
        if has_notices:
            notices = pd.read_csv(notices_file, parse_dates=["timestamp"])
            print(f"\n[3/5] 使用已有公告: {len(notices)} 条")
        else:
            notices = fetch_stock_notices_multithreaded()

        # 4. 获取宏观数据（如果不存在）
        if has_macro:
            print(f"\n[4/5] 使用已有宏观数据")
            macro = {
                "cpi": pd.read_csv(macro_cpi_file, parse_dates=["timestamp"]),
                "pmi": pd.read_csv(macro_pmi_file, parse_dates=["timestamp"]),
                "lpr": pd.read_csv(macro_lpr_file, parse_dates=["timestamp"]),
            }
        else:
            macro = fetch_macro_data()

        # 5. 获取北向资金（如果不存在）
        if has_northbound:
            print(f"\n[5/5] 使用已有北向资金")
            northbound = pd.read_csv(northbound_file, parse_dates=["timestamp"])
        else:
            northbound = fetch_northbound_flow()

        # 6. 转换宏观数据和北向资金为新闻格式
        macro_news = macro_to_news_format(macro)
        northbound_news = northbound_to_news_format(northbound)

        # 7. 合并所有信息源
        combined_news = create_combined_news(news, notices, macro_news, northbound_news)

        # 8. 验证数据
        validate_all_data(ohlcv, combined_news, macro)

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("数据获取完成！")
        print("=" * 60)
        print(f"总耗时: {elapsed/60:.1f} 分钟")
        print("\n数据文件汇总 (CSV格式，兼容回测框架):")
        print(f"  📊 OHLCV:       data/raw/real_csi300_5y.csv")
        print(f"  📰 CCTV新闻:    data/raw/csi300_news_cctv_2020_2024.csv")
        print(f"  📢 公告:        data/raw/csi300_notices_2020_2024.csv")
        print(f"  📰 合并新闻:    data/raw/csi300_news_combined_2020_2024.csv")
        print(f"  📈 CPI:         data/raw/macro_cpi_2020_2024.csv")
        print(f"  📈 PMI:         data/raw/macro_pmi_2020_2024.csv")
        print(f"  📈 LPR利率:     data/raw/macro_lpr_2020_2024.csv")
        print(f"  💰 北向资金:    data/raw/northbound_flow_2020_2024.csv")

        print("\n下一步:")
        print("  1. 构建LLM缓存:")
        print(f"     python main.py cache-build --symbol CSI300 \\")
        print(f"       --start {START_DATE} --end {END_DATE} \\")
        print(f"       --news-file data/raw/csi300_news_combined_2020_2024.csv")

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断，数据已分块保存，可重新运行继续")
    except Exception as e:
        print(f"\n\n✗ 数据获取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
