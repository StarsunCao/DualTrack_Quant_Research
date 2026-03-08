#!/usr/bin/env python
"""
清洗和融合多源新闻数据。

数据源：
1. Kaggle Financial News Headlines (2010-2020)
2. Hugging Face Twitter Financial News (2020-2023)
3. Hugging Face ESG News (2020-2023)

输出：
- data/raw/us_market_news/us_news_open_source_2010_2023.csv

使用方法：
    python scripts/clean_and_merge_news.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_kaggle_data():
    """加载 Kaggle 数据。"""
    print("\n" + "=" * 60)
    print("加载 Kaggle 数据...")
    print("=" * 60)

    kaggle_dir = project_root / "data" / "raw" / "kaggle_news"

    # 查找 CSV 文件
    csv_files = list(kaggle_dir.glob("*.csv"))

    if not csv_files:
        print("❌ 未找到 Kaggle 数据文件")
        print(f"   路径: {kaggle_dir}")
        return None

    # 尝试读取文件
    for csv_file in csv_files:
        try:
            print(f"  尝试读取: {csv_file.name}")
            df = pd.read_csv(csv_file)

            # 显示原始列名
            print(f"  原始列名: {df.columns.tolist()}")

            # 检查必要的列
            # Kaggle 数据集可能的列名
            possible_time_cols = ["time", "date", "datetime", "timestamp"]
            possible_title_cols = ["headline", "title", "news", "text"]

            # 查找匹配的列
            time_col = None
            title_col = None

            for col in possible_time_cols:
                if col in df.columns:
                    time_col = col
                    break

            for col in possible_title_cols:
                if col in df.columns:
                    title_col = col
                    break

            if time_col is None or title_col is None:
                print(f"  ⚠️  跳过此文件（缺少必要列）")
                continue

            # 标准化列名
            df = df.rename(columns={time_col: "timestamp", title_col: "title"})

            # 如果没有 content 列，使用 title 作为 content
            if "content" not in df.columns:
                df["content"] = df["title"]

            # 添加数据源标记
            df["source"] = "kaggle_headlines"

            print(f"  ✅ 成功加载: {len(df):,} 条")
            return df

        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            continue

    return None


def load_twitter_data():
    """加载 Twitter 数据。"""
    print("\n" + "=" * 60)
    print("加载 Twitter 数据...")
    print("=" * 60)

    twitter_file = (
        project_root / "data" / "raw" / "huggingface" / "twitter_financial_news.csv"
    )

    if not twitter_file.exists():
        print("❌ Twitter 数据文件不存在")
        print(f"   路径: {twitter_file}")
        return None

    try:
        df = pd.read_csv(twitter_file)
        print(f"  原始列名: {df.columns.tolist()}")

        # 标准化列名
        column_mapping = {"text": "content"}
        df = df.rename(columns=column_mapping)

        # 如果没有 title 列，使用 content 的前 100 个字符
        if "title" not in df.columns:
            df["title"] = df["content"].astype(str).str[:100]

        # 如果没有 timestamp 列，尝试其他可能的列
        if "timestamp" not in df.columns:
            possible_cols = ["date", "time", "created_at", "datetime"]
            for col in possible_cols:
                if col in df.columns:
                    df = df.rename(columns={col: "timestamp"})
                    break

        # 添加数据源标记
        df["source"] = "huggingface_twitter"

        print(f"  ✅ 成功加载: {len(df):,} 条")
        return df

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None


def load_esg_data():
    """加载 ESG 数据。"""
    print("\n" + "=" * 60)
    print("加载 ESG 数据...")
    print("=" * 60)

    esg_file = project_root / "data" / "raw" / "huggingface" / "esg_news.csv"

    if not esg_file.exists():
        print("❌ ESG 数据文件不存在")
        print(f"   路径: {esg_file}")
        return None

    try:
        df = pd.read_csv(esg_file)
        print(f"  原始列名: {df.columns.tolist()}")

        # 标准化列名
        column_mapping = {"headline": "title", "news": "content"}
        df = df.rename(columns=column_mapping)

        # 如果没有 content 列，使用 title
        if "content" not in df.columns:
            df["content"] = df["title"]

        # 添加数据源标记
        df["source"] = "huggingface_esg"

        print(f"  ✅ 成功加载: {len(df):,} 条")
        return df

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None


def clean_and_validate(df, source_name):
    """数据清洗和验证。"""
    print(f"\n清洗 {source_name} 数据:")
    print(f"  原始数据量: {len(df):,} 条")

    # 1. 时间戳处理
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        before_drop = len(df)
        df = df.dropna(subset=["timestamp"])
        after_drop = len(df)
        if before_drop != after_drop:
            print(f"  移除无效时间戳: {before_drop - after_drop:,} 条")

    # 2. 内容清洗
    if "content" not in df.columns:
        df["content"] = df["title"]

    # 3. 最小长度过滤（至少 50 个字符）
    before_filter = len(df)
    df = df[df["content"].astype(str).str.len() >= 50]
    after_filter = len(df)
    if before_filter != after_filter:
        print(f"  过滤短内容: {before_filter - after_filter:,} 条")

    # 4. 去重
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["title"])
    after_dedup = len(df)
    if before_dedup != after_dedup:
        print(f"  去重: {before_dedup - after_dedup:,} 条")

    # 5. 排序
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  清洗后数据量: {len(df):,} 条")

    return df


def add_synthetic_timestamp(df):
    """为缺少时间戳的数据添加合成时间戳（用于测试）。"""
    if "timestamp" not in df.columns or df["timestamp"].isnull().all():
        print("\n  添加合成时间戳（2020-2023 随机分布）...")
        import numpy as np

        # 生成 2020-2023 年的随机时间戳
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2023-12-31")

        # 计算时间范围（秒）
        date_range = (end_date - start_date).total_seconds()

        # 生成随机偏移量
        random_offsets = np.random.uniform(0, date_range, size=len(df))

        # 创建时间戳
        df["timestamp"] = start_date + pd.to_timedelta(random_offsets, unit="s")

        print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")

    return df


def merge_all_datasets():
    """合并所有数据集。"""
    print("=" * 60)
    print("开始加载和清洗数据集...")
    print("=" * 60)

    all_data = []

    # 加载各数据集
    kaggle_df = load_kaggle_data()
    if kaggle_df is not None:
        kaggle_df = clean_and_validate(kaggle_df, "Kaggle")
        all_data.append(kaggle_df)

    twitter_df = load_twitter_data()
    if twitter_df is not None:
        twitter_df = clean_and_validate(twitter_df, "Twitter")
        all_data.append(twitter_df)

    esg_df = load_esg_data()
    if esg_df is not None:
        esg_df = clean_and_validate(esg_df, "ESG")
        all_data.append(esg_df)

    if not all_data:
        print("\n❌ 没有可用数据")
        return None

    # 合并
    print("\n" + "=" * 60)
    print("合并数据集...")
    print("=" * 60)

    combined = pd.concat(all_data, ignore_index=True)

    # 最终去重
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["title"])
    after_dedup = len(combined)
    print(f"最终去重: {before_dedup - after_dedup:,} 条")

    # 添加合成时间戳（如果需要）
    combined = add_synthetic_timestamp(combined)

    # 排序
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").reset_index(drop=True)

    # 确保必要的列存在
    required_cols = ["timestamp", "title", "content", "source"]
    for col in required_cols:
        if col not in combined.columns:
            print(f"⚠️  缺少列: {col}")

    # 只保留必要的列
    available_cols = [col for col in required_cols if col in combined.columns]
    combined = combined[available_cols]

    # 保存
    output_dir = project_root / "data" / "raw" / "us_market_news"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "us_news_open_source_2010_2023.csv"
    combined.to_csv(output_file, index=False)

    print(f"\n✅ 合并完成:")
    print(f"   总数据量: {len(combined):,} 条")

    if "timestamp" in combined.columns and len(combined) > 0:
        print(
            f"   时间范围: {combined['timestamp'].min()} 至 {combined['timestamp'].max()}"
        )

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

    if "timestamp" in df.columns and len(df) > 0:
        print(
            f"时间范围: {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')}"
        )

    if "content" in df.columns:
        print(f"平均内容长度: {df['content'].astype(str).str.len().mean():.0f} 字符")

    # 数据源分布
    if "source" in df.columns:
        print(f"\n数据源分布:")
        print(df["source"].value_counts())

    # 年度分布
    if "timestamp" in df.columns and len(df) > 0:
        df["year"] = df["timestamp"].dt.year
        print(f"\n年度分布:")
        year_counts = df["year"].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {int(year)}: {count:,} 条")

    # 缺失值检查
    print(f"\n缺失值检查:")
    print(df.isnull().sum())


def main():
    """主函数。"""
    df = merge_all_datasets()

    if df is not None:
        generate_report(df)
        print("\n" + "=" * 60)
        print("🎉 数据处理完成！")
        print("=" * 60)
        print("\n下一步：验证数据质量")
        print("   python tests/test_us_news_data.py")
    else:
        print("\n" + "=" * 60)
        print("❌ 数据处理失败")
        print("=" * 60)
        print("\n请确保已运行数据下载脚本:")
        print("   python scripts/download_open_source_news.py")


if __name__ == "__main__":
    main()