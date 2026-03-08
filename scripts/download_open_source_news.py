#!/usr/bin/env python
"""
下载开源美股新闻数据集。

数据源：
1. Kaggle: Financial News Headlines (2010-2020)
2. Hugging Face: Twitter Financial News (2020-2023)
3. Hugging Face: ESG News (2020-2023)

使用方法：
    python scripts/download_open_source_news.py

注意：
    - Kaggle 数据集需要先配置 API Token
    - Hugging Face 数据集会自动下载
"""

import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import subprocess

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_kaggle_config():
    """检查 Kaggle API 配置。"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print("=" * 60)
        print("❌ Kaggle API 未配置")
        print("=" * 60)
        print("\n请按以下步骤配置：")
        print("1. 注册 Kaggle 账号：https://www.kaggle.com/")
        print("2. 进入 My Account -> API -> Create New API Token")
        print("3. 下载 kaggle.json 文件")
        print("4. 执行以下命令：")
        print("   mkdir -p ~/.kaggle")
        print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        print()
        return False

    return True


def download_kaggle_headlines():
    """下载 Kaggle 金融新闻标题数据集。"""
    print("\n" + "=" * 60)
    print("下载 Kaggle Financial News Headlines (2010-2020)...")
    print("=" * 60)

    output_dir = project_root / "data" / "raw" / "kaggle_news"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 使用 Kaggle CLI 下载
        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "notlucasp/financial-news-headlines",
                "-p",
                str(output_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Kaggle 数据集下载成功")
            print(f"   保存路径: {output_dir}")

            # 列出下载的文件
            files = list(output_dir.glob("*.csv"))
            print(f"   文件列表:")
            for f in files:
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"     - {f.name} ({size_mb:.2f} MB)")

            return True
        else:
            print(f"❌ Kaggle 下载失败:")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print("❌ Kaggle CLI 未安装")
        print("   请运行: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        return False


def download_huggingface_twitter():
    """下载 Hugging Face Twitter 金融新闻数据集。"""
    print("\n" + "=" * 60)
    print("下载 Hugging Face Twitter Financial News (2020-2023)...")
    print("=" * 60)

    output_dir = project_root / "data" / "raw" / "huggingface"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("  正在从 Hugging Face Hub 下载...")

        # 下载数据集
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

        # 转换为 DataFrame
        df = pd.DataFrame(dataset["train"])

        # 保存
        output_file = output_dir / "twitter_financial_news.csv"
        df.to_csv(output_file, index=False)

        print(f"✅ Twitter 数据集下载成功")
        print(f"   数据量: {len(df):,} 条")
        print(f"   保存路径: {output_file}")
        print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Twitter 数据集下载失败: {e}")
        return False


def download_huggingface_esg():
    """下载 Hugging Face ESG 新闻数据集。"""
    print("\n" + "=" * 60)
    print("下载 Hugging Face ESG News (2020-2023)...")
    print("=" * 60)

    output_dir = project_root / "data" / "raw" / "huggingface"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("  正在从 Hugging Face Hub 下载...")

        # 下载数据集
        dataset = load_dataset("TheFinAI/esg-news")

        # 转换为 DataFrame
        df = pd.DataFrame(dataset["train"])

        # 保存
        output_file = output_dir / "esg_news.csv"
        df.to_csv(output_file, index=False)

        print(f"✅ ESG 数据集下载成功")
        print(f"   数据量: {len(df):,} 条")
        print(f"   保存路径: {output_file}")
        print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ ESG 数据集下载失败: {e}")
        return False


def main():
    """主函数：下载所有数据集。"""
    print("=" * 60)
    print("开源美股新闻数据集下载工具")
    print("=" * 60)

    # 检查 Kaggle 配置
    if not check_kaggle_config():
        print("\n⚠️  跳过 Kaggle 数据集下载")
        print("   您可以稍后手动下载并放置到 data/raw/kaggle_news/")
        kaggle_success = False
    else:
        kaggle_success = download_kaggle_headlines()

    # 下载 Hugging Face 数据集
    twitter_success = download_huggingface_twitter()
    esg_success = download_huggingface_esg()

    # 总结
    print("\n" + "=" * 60)
    print("下载总结")
    print("=" * 60)

    results = [
        ("Kaggle Financial News (2010-2020)", kaggle_success),
        ("HuggingFace Twitter News (2020-2023)", twitter_success),
        ("HuggingFace ESG News (2020-2023)", esg_success),
    ]

    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")

    # 统计
    success_count = sum([s for _, s in results])
    total_count = len(results)

    print()
    if success_count == total_count:
        print("🎉 所有数据集下载完成！")
        print("\n下一步：运行数据清洗脚本")
        print("   python scripts/clean_and_merge_news.py")
    elif success_count > 0:
        print(f"⚠️  部分数据集下载成功 ({success_count}/{total_count})")
        print("\n您可以继续运行清洗脚本处理已下载的数据")
        print("   python scripts/clean_and_merge_news.py")
    else:
        print("❌ 所有数据集下载失败")
        print("   请检查网络连接和配置")


if __name__ == "__main__":
    main()