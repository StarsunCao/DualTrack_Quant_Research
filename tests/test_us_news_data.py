#!/usr/bin/env python
"""
测试美股新闻数据质量。

测试内容：
1. 数据文件是否存在
2. 数据加载是否正常
3. 数据列是否完整
4. 数据质量检查（时间格式、内容长度、去重）

使用方法：
    python tests/test_us_news_data.py
"""

import unittest
from pathlib import Path
import pandas as pd


class TestUSNewsData(unittest.TestCase):
    """美股新闻数据测试类。"""

    def setUp(self):
        """测试初始化。"""
        self.data_file = Path(
            "data/raw/us_market_news/us_news_open_source_2010_2023.csv"
        )

    def test_data_file_exists(self):
        """测试数据文件是否存在。"""
        self.assertTrue(
            self.data_file.exists(),
            f"数据文件不存在: {self.data_file}\n请先运行: python scripts/clean_and_merge_news.py",
        )

    def test_data_loading(self):
        """测试数据加载。"""
        if not self.data_file.exists():
            self.skipTest("数据文件不存在")

        df = pd.read_csv(self.data_file)
        self.assertGreater(len(df), 0, "数据为空")

        # 检查数据量是否合理（至少 5,000 条）
        # 注：由于使用开源数据集，数据量可能少于预期
        self.assertGreater(
            len(df),
            5000,
            f"数据量过少: {len(df)} 条（预期至少 5,000 条）",
        )

    def test_data_columns(self):
        """测试数据列是否完整。"""
        if not self.data_file.exists():
            self.skipTest("数据文件不存在")

        df = pd.read_csv(self.data_file)
        required_cols = ["timestamp", "title", "content", "source"]

        for col in required_cols:
            self.assertIn(col, df.columns, f"缺少列: {col}")

    def test_timestamp_format(self):
        """测试时间戳格式。"""
        if not self.data_file.exists():
            self.skipTest("数据文件不存在")

        df = pd.read_csv(self.data_file)

        if "timestamp" not in df.columns:
            self.fail("缺少 timestamp 列")

        # 转换时间戳
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # 检查是否有无效时间戳
        null_count = df["timestamp"].isnull().sum()
        self.assertEqual(
            null_count,
            0,
            f"时间戳包含 {null_count} 个空值",
        )

    def test_content_quality(self):
        """测试内容质量。"""
        if not self.data_file.exists():
            self.skipTest("数据文件不存在")

        df = pd.read_csv(self.data_file)

        if "content" not in df.columns:
            self.fail("缺少 content 列")

        # 检查内容长度（至少 50 字符）
        short_content = (df["content"].astype(str).str.len() < 50).sum()
        self.assertEqual(
            short_content,
            0,
            f"有 {short_content} 条内容过短（< 50 字符）",
        )

    def test_no_duplicates(self):
        """测试数据去重。"""
        if not self.data_file.exists():
            self.skipTest("数据文件不存在")

        df = pd.read_csv(self.data_file)

        if "title" not in df.columns:
            self.fail("缺少 title 列")

        # 检查重复标题
        duplicates = df.duplicated(subset=["title"]).sum()
        self.assertEqual(
            duplicates,
            0,
            f"存在 {duplicates} 条重复数据",
        )

    def test_source_distribution(self):
        """测试数据源分布。"""
        if not self.data_file.exists():
            self.skipTest("数据文件不存在")

        df = pd.read_csv(self.data_file)

        if "source" not in df.columns:
            self.fail("缺少 source 列")

        # 检查数据源
        source_counts = df["source"].value_counts()
        print(f"\n数据源分布:")
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} 条")

        # 至少应该有一个数据源
        self.assertGreater(len(source_counts), 0, "没有数据源")

    def test_time_range(self):
        """测试时间范围。"""
        if not self.data_file.exists():
            self.skipTest("数据文件不存在")

        df = pd.read_csv(self.data_file)

        if "timestamp" not in df.columns:
            self.fail("缺少 timestamp 列")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        print(f"\n时间范围:")
        print(f"  起始: {min_date.strftime('%Y-%m-%d')}")
        print(f"  结束: {max_date.strftime('%Y-%m-%d')}")

        # 检查时间范围是否合理（至少跨越 1 年）
        time_span = (max_date - min_date).days
        self.assertGreater(
            time_span,
            365,
            f"时间跨度过短: {time_span} 天（预期至少 365 天）",
        )


def run_tests():
    """运行所有测试。"""
    print("=" * 60)
    print("美股新闻数据质量测试")
    print("=" * 60)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUSNewsData)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ 所有测试通过！数据质量良好。")
        print("\n下一步：集成到回测系统")
        print("   python main.py run --symbol QQQ --track deepseek-v3.2")
    else:
        print("\n❌ 部分测试失败，请检查数据质量。")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()