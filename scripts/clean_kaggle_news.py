#!/usr/bin/env python
"""
清洗和融合 Kaggle 金融新闻数据（包含真实时间戳）。

数据源：
1. CNBC Headlines (2020, 3,080 条)
2. Reuters Headlines (2020, 32,770 条)
3. Guardian Headlines (2020, 17,800 条)

输出：
- data/raw/us_market_news/us_news_real_timestamp_2020.csv

使用方法：
    python scripts/clean_kaggle_news.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import re

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_cnbc_time(time_str):
    """
    解析 CNBC 时间格式。

    示例: " 7:51  PM ET Fri, 17 July 2020"
    """
    try:
        # 清理字符串
        time_str = str(time_str).strip()

        # 提取日期部分
        # 格式: "7:51 PM ET Fri, 17 July 2020"
        match = re.search(r'(\w+),\s+(\d+)\s+(\w+)\s+(\d{4})', time_str)
        if match:
            day_name, day, month, year = match.groups()
            date_str = f"{day} {month} {year}"
            return pd.to_datetime(date_str, format='%d %B %Y')

        return pd.NaT
    except:
        return pd.NaT


def parse_reuters_time(time_str):
    """
    解析 Reuters 时间格式。

    示例: "Jul 18 2020"
    """
    try:
        return pd.to_datetime(str(time_str).strip(), format='%b %d %Y')
    except:
        return pd.NaT


def parse_guardian_time(time_str):
    """
    解析 Guardian 时间格式。

    示例: "18-Jul-20"
    """
    try:
        return pd.to_datetime(str(time_str).strip(), format='%d-%b-%y')
    except:
        return pd.NaT


def load_cnbc_data():
    """加载 CNBC 数据。"""
    print("\n" + "=" * 60)
    print("加载 CNBC 数据...")
    print("=" * 60)

    file_path = project_root / "data" / "raw" / "kaggle_news" / "cnbc_headlines.csv"

    if not file_path.exists():
        print("❌ CNBC 数据文件不存在")
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

        # 检查解析成功率
        success_rate = df['timestamp'].notna().sum() / len(df) * 100
        print(f"  时间戳解析成功: {success_rate:.1f}%")

        # 移除解析失败的记录
        df = df.dropna(subset=['timestamp'])

        # 添加数据源标记
        df['source'] = 'cnbc'

        print(f"  有效数据: {len(df):,} 条")
        print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")

        return df[['timestamp', 'title', 'content', 'source']]

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None


def load_reuters_data():
    """加载 Reuters 数据。"""
    print("\n" + "=" * 60)
    print("加载 Reuters 数据...")
    print("=" * 60)

    file_path = project_root / "data" / "raw" / "kaggle_news" / "reuters_headlines.csv"

    if not file_path.exists():
        print("❌ Reuters 数据文件不存在")
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

        # 检查解析成功率
        success_rate = df['timestamp'].notna().sum() / len(df) * 100
        print(f"  时间戳解析成功: {success_rate:.1f}%")

        # 移除解析失败的记录
        df = df.dropna(subset=['timestamp'])

        # 添加数据源标记
        df['source'] = 'reuters'

        print(f"  有效数据: {len(df):,} 条")
        print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")

        return df[['timestamp', 'title', 'content', 'source']]

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None


def load_guardian_data():
    """加载 Guardian 数据。"""
    print("\n" + "=" * 60)
    print("加载 Guardian 数据...")
    print("=" * 60)

    file_path = project_root / "data" / "raw" / "kaggle_news" / "guardian_headlines.csv"

    if not file_path.exists():
        print("❌ Guardian 数据文件不存在")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"  原始数据: {len(df):,} 条")

        # 重命名列
        df = df.rename(columns={
            'Headlines': 'title',
            'Time': 'timestamp_raw'
        })

        # Guardian 数据没有 content，使用 title 作为 content
        df['content'] = df['title']

        # 解析时间戳
        print("  解析时间戳...")
        df['timestamp'] = df['timestamp_raw'].apply(parse_guardian_time)

        # 检查解析成功率
        success_rate = df['timestamp'].notna().sum() / len(df) * 100
        print(f"  时间戳解析成功: {success_rate:.1f}%")

        # 移除解析失败的记录
        df = df.dropna(subset=['timestamp'])

        # 添加数据源标记
        df['source'] = 'guardian'

        print(f"  有效数据: {len(df):,} 条")
        print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")

        return df[['timestamp', 'title', 'content', 'source']]

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None


def clean_and_validate(df, source_name):
    """数据清洗和验证。"""
    print(f"\n清洗 {source_name} 数据:")
    print(f"  原始数据量: {len(df):,} 条")

    # 1. 移除空标题
    before = len(df)
    df = df.dropna(subset=['title'])
    after = len(df)
    if before != after:
        print(f"  移除空标题: {before - after:,} 条")

    # 2. 过滤短内容（至少 30 字符）
    before = len(df)
    df = df[df['content'].astype(str).str.len() >= 30]
    after = len(df)
    if before != after:
        print(f"  过滤短内容: {before - after:,} 条")

    # 3. 去重
    before = len(df)
    df = df.drop_duplicates(subset=['title'])
    after = len(df)
    if before != after:
        print(f"  去重: {before - after:,} 条")

    print(f"  清洗后数据量: {len(df):,} 条")

    return df


def merge_all_datasets():
    """合并所有数据集。"""
    print("=" * 60)
    print("开始加载和清洗 Kaggle 新闻数据...")
    print("=" * 60)

    all_data = []

    # 加载各数据集
    cnbc_df = load_cnbc_data()
    if cnbc_df is not None:
        cnbc_df = clean_and_validate(cnbc_df, "CNBC")
        all_data.append(cnbc_df)

    reuters_df = load_reuters_data()
    if reuters_df is not None:
        reuters_df = clean_and_validate(reuters_df, "Reuters")
        all_data.append(reuters_df)

    guardian_df = load_guardian_data()
    if guardian_df is not None:
        guardian_df = clean_and_validate(guardian_df, "Guardian")
        all_data.append(guardian_df)

    if not all_data:
        print("\n❌ 没有可用数据")
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

    # 排序
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # 保存
    output_dir = project_root / "data" / "raw" / "us_market_news"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "us_news_real_timestamp_2020.csv"
    combined.to_csv(output_file, index=False)

    print(f"\n✅ 合并完成:")
    print(f"   总数据量: {len(combined):,} 条")
    print(f"   时间范围: {combined['timestamp'].min()} 至 {combined['timestamp'].max()}")
    print(f"   文件路径: {output_file}")
    print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return combined


def generate_report(df):
    """生成数据质量报告。"""
    print("\n" + "=" * 60)
    print("数据质量报告")
    print("=" * 60)

    # 基本信息
    print(f"\n总数据量: {len(df):,} 条")
    print(f"时间范围: {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"平均内容长度: {df['content'].astype(str).str.len().mean():.0f} 字符")

    # 数据源分布
    print(f"\n数据源分布:")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        percentage = count / len(df) * 100
        print(f"  {source}: {count:,} 条 ({percentage:.1f}%)")

    # 月度分布
    df['month'] = df['timestamp'].dt.to_period('M')
    print(f"\n月度分布:")
    month_counts = df['month'].value_counts().sort_index()
    for month, count in month_counts.items():
        print(f"  {month}: {count:,} 条")

    # 缺失值检查
    print(f"\n缺失值检查:")
    print(df.isnull().sum())

    # 时间戳完整性
    print(f"\n时间戳完整性:")
    print(f"  有效时间戳: {df['timestamp'].notna().sum():,} 条")
    print(f"  缺失时间戳: {df['timestamp'].isna().sum():,} 条")


def main():
    """主函数。"""
    df = merge_all_datasets()

    if df is not None:
        generate_report(df)

        print("\n" + "=" * 60)
        print("🎉 数据处理完成！")
        print("=" * 60)
        print("\n✅ 所有数据包含真实时间戳")
        print("\n下一步：验证数据质量")
        print("   python tests/test_us_news_data.py")
    else:
        print("\n" + "=" * 60)
        print("❌ 数据处理失败")
        print("=" * 60)


if __name__ == "__main__":
    main()