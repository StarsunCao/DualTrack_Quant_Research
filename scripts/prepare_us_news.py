#!/usr/bin/env python
"""
准备美股新闻数据集（用于 QQQ 回测）。

根据计划文档，此脚本：
1. 只使用 Reuters 和 CNBC 数据（排除 Guardian 英国新闻）
2. 过滤时间范围: 2018-01-02 至 2020-07-17
3. 输出统一格式: [timestamp, title, content, source]

使用方法：
    python scripts/prepare_us_news.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import re

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 目标时间范围（QQQ 测试期）
START_DATE = pd.Timestamp('2018-01-02')
END_DATE = pd.Timestamp('2020-07-17')


def parse_cnbc_time(time_str: str) -> pd.Timestamp:
    """
    解析 CNBC 时间格式。

    示例: " 7:51  PM ET Fri, 17 July 2020"
    """
    try:
        time_str = str(time_str).strip()

        # 格式: "7:51 PM ET Fri, 17 July 2020"
        match = re.search(r'(\w+),\s+(\d+)\s+(\w+)\s+(\d{4})', time_str)
        if match:
            day_name, day, month, year = match.groups()
            date_str = f"{day} {month} {year}"
            return pd.to_datetime(date_str, format='%d %B %Y')

        return pd.NaT
    except Exception:
        return pd.NaT


def parse_reuters_time(time_str: str) -> pd.Timestamp:
    """
    解析 Reuters 时间格式。

    示例: "Jul 18 2020"
    """
    try:
        return pd.to_datetime(str(time_str).strip(), format='%b %d %Y')
    except Exception:
        return pd.NaT


def load_cnbc_data() -> pd.DataFrame | None:
    """加载 CNBC 数据。"""
    print("\n" + "=" * 60)
    print("加载 CNBC 数据...")
    print("=" * 60)

    file_path = project_root / "data" / "raw" / "kaggle_news" / "cnbc_headlines.csv"

    if not file_path.exists():
        print("  CNBC 数据文件不存在")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"  原始数据: {len(df):,} 条")

        # 重命名列
        df = df.rename(columns={
            'Headlines': 'title',
            'Time': 'timestamp_raw',
            'Description': 'content'
        })

        # 解析时间戳
        print("  解析时间戳...")
        df['timestamp'] = df['timestamp_raw'].apply(parse_cnbc_time)

        # 移除解析失败的记录
        df = df.dropna(subset=['timestamp'])

        # 添加数据源标记
        df['source'] = 'cnbc'

        print(f"  有效数据: {len(df):,} 条")
        if len(df) > 0:
            print(f"  时间范围: {df['timestamp'].min().date()} 至 {df['timestamp'].max().date()}")

        return df[['timestamp', 'title', 'content', 'source']]

    except Exception as e:
        print(f"  加载失败: {e}")
        return None


def load_reuters_data() -> pd.DataFrame | None:
    """加载 Reuters 数据。"""
    print("\n" + "=" * 60)
    print("加载 Reuters 数据...")
    print("=" * 60)

    file_path = project_root / "data" / "raw" / "kaggle_news" / "reuters_headlines.csv"

    if not file_path.exists():
        print("  Reuters 数据文件不存在")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"  原始数据: {len(df):,} 条")

        # 重命名列
        df = df.rename(columns={
            'Headlines': 'title',
            'Time': 'timestamp_raw',
            'Description': 'content'
        })

        # 解析时间戳
        print("  解析时间戳...")
        df['timestamp'] = df['timestamp_raw'].apply(parse_reuters_time)

        # 移除解析失败的记录
        df = df.dropna(subset=['timestamp'])

        # 添加数据源标记
        df['source'] = 'reuters'

        print(f"  有效数据: {len(df):,} 条")
        if len(df) > 0:
            print(f"  时间范围: {df['timestamp'].min().date()} 至 {df['timestamp'].max().date()}")

        return df[['timestamp', 'title', 'content', 'source']]

    except Exception as e:
        print(f"  加载失败: {e}")
        return None


def clean_data(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """数据清洗。"""
    print(f"\n清洗 {source_name} 数据:")
    print(f"  原始数据量: {len(df):,} 条")

    # 1. 移除空标题
    before = len(df)
    df = df.dropna(subset=['title'])
    df = df[df['title'].astype(str).str.strip() != '']
    after = len(df)
    if before != after:
        print(f"  移除空标题: {before - after:,} 条")

    # 2. 过滤短内容（至少 30 字符，保证有足够信息）
    before = len(df)
    df = df[df['content'].astype(str).str.len() >= 30]
    after = len(df)
    if before != after:
        print(f"  过滤短内容(<30字符): {before - after:,} 条")

    # 3. 去重（基于标题）
    before = len(df)
    df = df.drop_duplicates(subset=['title'])
    after = len(df)
    if before != after:
        print(f"  去重: {before - after:,} 条")

    print(f"  清洗后数据量: {len(df):,} 条")

    return df


def filter_by_date_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """过滤时间范围。"""
    print(f"\n过滤时间范围: {start.date()} 至 {end.date()}")

    before = len(df)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    after = len(df)

    print(f"  过滤掉: {before - after:,} 条")
    print(f"  保留: {after:,} 条")

    return df


def merge_datasets() -> pd.DataFrame | None:
    """合并数据集。"""
    print("=" * 60)
    print("准备美股新闻数据集")
    print("=" * 60)
    print(f"目标时间范围: {START_DATE.date()} 至 {END_DATE.date()}")

    all_data = []

    # 加载 Reuters（美股高质量数据）
    reuters_df = load_reuters_data()
    if reuters_df is not None:
        reuters_df = clean_data(reuters_df, "Reuters")
        all_data.append(reuters_df)

    # 加载 CNBC（美股高质量数据）
    cnbc_df = load_cnbc_data()
    if cnbc_df is not None:
        cnbc_df = clean_data(cnbc_df, "CNBC")
        all_data.append(cnbc_df)

    if not all_data:
        print("\n没有可用数据")
        return None

    # 合并
    print("\n" + "=" * 60)
    print("合并数据集...")
    print("=" * 60)

    combined = pd.concat(all_data, ignore_index=True)

    # 最终去重
    before = len(combined)
    combined = combined.drop_duplicates(subset=['title'])
    after = len(combined)
    print(f"最终去重: {before - after:,} 条")

    # 过滤时间范围
    combined = filter_by_date_range(combined, START_DATE, END_DATE)

    if len(combined) == 0:
        print("\n过滤后无数据")
        return None

    # 排序
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # 保存
    output_dir = project_root / "data" / "raw" / "us_market_news"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "us_news_prepared.csv"
    combined.to_csv(output_file, index=False)

    print(f"\n保存完成:")
    print(f"   总数据量: {len(combined):,} 条")
    print(f"   时间范围: {combined['timestamp'].min().date()} 至 {combined['timestamp'].max().date()}")
    print(f"   文件路径: {output_file}")
    print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return combined


def generate_report(df: pd.DataFrame) -> None:
    """生成数据质量报告。"""
    print("\n" + "=" * 60)
    print("数据质量报告")
    print("=" * 60)

    # 基本信息
    print(f"\n总数据量: {len(df):,} 条")
    print(f"时间范围: {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"平均内容长度: {df['content'].astype(str).str.len().mean():.0f} 字符")
    print(f"平均标题长度: {df['title'].astype(str).str.len().mean():.0f} 字符")

    # 数据源分布
    print(f"\n数据源分布:")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        percentage = count / len(df) * 100
        print(f"  {source}: {count:,} 条 ({percentage:.1f}%)")

    # 年度分布
    df['year'] = df['timestamp'].dt.year
    print(f"\n年度分布:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} 条")

    # 月度分布
    df['month'] = df['timestamp'].dt.to_period('M')
    print(f"\n月度分布（前 10 个月）:")
    month_counts = df['month'].value_counts().sort_index()
    for i, (month, count) in enumerate(month_counts.items()):
        if i >= 10:
            print(f"  ... 还有 {len(month_counts) - 10} 个月")
            break
        print(f"  {month}: {count:,} 条")

    # 检查内容相关性（抽查关键词）
    print(f"\n内容相关性检查（关键词统计）:")
    keywords = ['stock', 'market', 'NASDAQ', 'S&P', 'Apple', 'Microsoft', 'Amazon', 'Google', 'Facebook', 'tech']
    content_lower = df['content'].astype(str).str.lower() + ' ' + df['title'].astype(str).str.lower()
    for kw in keywords:
        count = content_lower.str.contains(kw.lower(), na=False).sum()
        print(f"  '{kw}': {count:,} 条 ({count/len(df)*100:.1f}%)")


def main():
    """主函数。"""
    df = merge_datasets()

    if df is not None:
        generate_report(df)

        print("\n" + "=" * 60)
        print("数据处理完成！")
        print("=" * 60)
        print("\n下一步: 生成新的 LLM 缓存")
        print("   python main.py cache-build \\")
        print("       --symbol QQQ \\")
        print("       --start 2018-01-02 \\")
        print("       --end 2020-07-17 \\")
        print("       --news-file data/raw/us_market_news/us_news_prepared.csv \\")
        print("       --executor siliconflow \\")
        print("       --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    else:
        print("\n" + "=" * 60)
        print("数据处理失败")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()