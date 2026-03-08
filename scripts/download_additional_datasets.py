#!/usr/bin/env python
"""
下载额外的开源金融新闻数据集。

替代数据源（如果原计划数据集不可用）：
1. Financial PhraseBank（学术标注数据）
2. Twitter Financial News Sentiment（已下载）
"""

import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_financial_phrasebank():
    """下载 Financial PhraseBank 数据集（学术标注数据）。"""
    print("\n" + "=" * 60)
    print("下载 Financial PhraseBank (学术标注数据)...")
    print("=" * 60)

    output_dir = project_root / "data" / "raw" / "huggingface"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("  正在从 Hugging Face Hub 下载...")

        # 下载数据集
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")

        # 转换为 DataFrame
        df = pd.DataFrame(dataset["train"])

        # 保存
        output_file = output_dir / "financial_phrasebank.csv"
        df.to_csv(output_file, index=False)

        print(f"✅ Financial PhraseBank 下载成功")
        print(f"   数据量: {len(df):,} 条")
        print(f"   保存路径: {output_file}")
        print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Financial PhraseBank 下载失败: {e}")
        return False


def download_fiqa_sentiment():
    """下载 FiQA 金融情感数据集。"""
    print("\n" + "=" * 60)
    print("下载 FiQA Financial Sentiment...")
    print("=" * 60)

    output_dir = project_root / "data" / "raw" / "huggingface"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("  正在从 Hugging Face Hub 下载...")

        # 下载数据集
        dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")

        # 转换为 DataFrame
        df = pd.DataFrame(dataset["train"])

        # 保存
        output_file = output_dir / "fiqa_sentiment.csv"
        df.to_csv(output_file, index=False)

        print(f"✅ FiQA Sentiment 下载成功")
        print(f"   数据量: {len(df):,} 条")
        print(f"   保存路径: {output_file}")
        print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ FiQA Sentiment 下载失败: {e}")
        return False


def main():
    """主函数。"""
    print("=" * 60)
    print("下载额外开源金融数据集")
    print("=" * 60)

    # 尝试下载多个数据集
    results = []

    results.append(("Financial PhraseBank", download_financial_phrasebank()))
    results.append(("FiQA Sentiment", download_fiqa_sentiment()))

    # 总结
    print("\n" + "=" * 60)
    print("下载总结")
    print("=" * 60)

    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()