"""
自动化报告生成模块。

一键生成完整实验报告，包含：
- 实验配置摘要
- 金融指标对比表
- 工程指标对比表
- 关键图表嵌入
- LaTeX 表格导出

输出格式：
- Markdown 报告
- CSV 数据表格
- LaTeX 表格
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    报告生成器。

    支持生成多种格式的实验报告。
    """

    def __init__(self, output_dir: str = "docs/output/reports"):
        """
        初始化报告生成器。

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        experiment_results: dict[str, Any],
        config: Optional[dict] = None,
        format: str = "all",
    ) -> dict[str, Path]:
        """
        生成完整实验报告。

        Args:
            experiment_results: 实验结果字典
            config: 实验配置
            format: 输出格式 ('markdown', 'csv', 'latex', 'all')

        Returns:
            生成的报告文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files = {}

        logger.info("开始生成实验报告...")

        if format in ["markdown", "all"]:
            md_path = self._generate_markdown_report(
                experiment_results, config, timestamp
            )
            generated_files["markdown"] = md_path

        if format in ["csv", "all"]:
            csv_path = self._generate_csv_tables(experiment_results, timestamp)
            generated_files["csv"] = csv_path

        if format in ["latex", "all"]:
            latex_path = self._generate_latex_tables(experiment_results, timestamp)
            generated_files["latex"] = latex_path

        if format in ["json", "all"]:
            json_path = self._generate_json_report(
                experiment_results, config, timestamp
            )
            generated_files["json"] = json_path

        logger.info(f"报告生成完成: {generated_files}")
        return generated_files

    def _generate_markdown_report(
        self,
        results: dict[str, Any],
        config: Optional[dict],
        timestamp: str,
    ) -> Path:
        """生成 Markdown 格式报告。"""
        report_path = self.output_dir / f"experiment_report_{timestamp}.md"

        lines = []

        # 标题
        lines.append("# DualTrack Quant Research - 实验报告\n")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. 实验配置摘要
        lines.append("## 1. 实验配置摘要\n")
        if config:
            lines.append("### 1.1 基本配置")
            lines.append("```yaml")
            lines.append(json.dumps(config, indent=2, ensure_ascii=False))
            lines.append("```\n")
        else:
            lines.append("*未提供配置信息*\n")

        # 2. 金融指标对比表
        lines.append("## 2. 金融指标对比\n")

        if "financial_metrics" in results:
            metrics_df = results["financial_metrics"]
            lines.append(metrics_df.to_markdown())
            lines.append("")

            # 添加分析
            best_sharpe = metrics_df["sharpe_ratio"].idxmax()
            best_return = metrics_df["total_return"].idxmax()
            best_drawdown = metrics_df["max_drawdown"].idxmin()

            lines.append("### 关键发现")
            lines.append(f"- **最高夏普比率**: {best_sharpe} ({metrics_df.loc[best_sharpe, 'sharpe_ratio']:.4f})")
            lines.append(f"- **最高总收益**: {best_return} ({metrics_df.loc[best_return, 'total_return']:.2%})")
            lines.append(f"- **最低最大回撤**: {best_drawdown} ({metrics_df.loc[best_drawdown, 'max_drawdown']:.2%})")
            lines.append("")
        else:
            lines.append("*无金融指标数据*\n")

        # 3. 工程指标对比表
        lines.append("## 3. 工程指标对比\n")

        if "engineering_metrics" in results:
            eng_df = results["engineering_metrics"]
            lines.append(eng_df.to_markdown())
            lines.append("")
        else:
            lines.append("*无工程指标数据*\n")

        # 4. 核心结论
        lines.append("## 4. 核心结论\n")

        if "conclusions" in results:
            for conclusion in results["conclusions"]:
                lines.append(f"- {conclusion}")
        else:
            lines.append("*待补充*")
        lines.append("")

        # 5. 图表索引
        lines.append("## 5. 图表索引\n")

        if "figures" in results:
            lines.append("| 图表 | 描述 | 路径 |")
            lines.append("|------|------|------|")
            for name, path in results["figures"].items():
                lines.append(f"| {name} | - | {path} |")
        else:
            lines.append("*无图表数据*")
        lines.append("")

        # 6. 附录：详细数据
        lines.append("## 附录：详细数据\n")

        if "raw_data" in results:
            lines.append("### 原始数据摘要")
            lines.append("```")
            for key, value in results["raw_data"].items():
                lines.append(f"{key}: {value}")
            lines.append("```")

        # 写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown 报告已生成: {report_path}")
        return report_path

    def _generate_csv_tables(
        self,
        results: dict[str, Any],
        timestamp: str,
    ) -> Path:
        """生成 CSV 格式表格。"""
        csv_dir = self.output_dir / f"csv_tables_{timestamp}"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # 金融指标表
        if "financial_metrics" in results:
            fm_path = csv_dir / "financial_metrics.csv"
            results["financial_metrics"].to_csv(fm_path)
            logger.debug(f"金融指标表已保存: {fm_path}")

        # 工程指标表
        if "engineering_metrics" in results:
            em_path = csv_dir / "engineering_metrics.csv"
            results["engineering_metrics"].to_csv(em_path)
            logger.debug(f"工程指标表已保存: {em_path}")

        # 交易记录
        if "trades" in results:
            trades_path = csv_dir / "trades.csv"
            results["trades"].to_csv(trades_path, index=False)
            logger.debug(f"交易记录已保存: {trades_path}")

        # 持仓记录
        if "positions" in results:
            pos_path = csv_dir / "positions.csv"
            results["positions"].to_csv(pos_path, index=False)
            logger.debug(f"持仓记录已保存: {pos_path}")

        logger.info(f"CSV 表格已生成: {csv_dir}")
        return csv_dir

    def _generate_latex_tables(
        self,
        results: dict[str, Any],
        timestamp: str,
    ) -> Path:
        """生成 LaTeX 格式表格（可直接用于论文）。"""
        latex_path = self.output_dir / f"tables_{timestamp}.tex"

        lines = []

        lines.append("% DualTrack Quant Research - 实验结果表格")
        lines.append(f"% 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 金融指标表
        if "financial_metrics" in results:
            df = results["financial_metrics"]

            lines.append("% 金融指标对比表")
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            lines.append("\\caption{金融指标对比}")
            lines.append("\\label{tab:financial_metrics}")

            # 构建表格列定义
            num_cols = len(df.columns)
            lines.append(f"\\begin{{tabular}}{{l{'c' * num_cols}}}")
            lines.append("\\toprule")

            # 表头
            header = "指标 & " + " & ".join(df.columns) + " \\\\"
            lines.append(header)
            lines.append("\\midrule")

            # 数据行
            for idx, row in df.iterrows():
                values = []
                for val in row.values:
                    if isinstance(val, float):
                        if abs(val) < 1:  # 百分比格式
                            values.append(f"{val:.2%}")
                        else:
                            values.append(f"{val:.4f}")
                    else:
                        values.append(str(val))

                row_str = f"{idx} & " + " & ".join(values) + " \\\\"
                lines.append(row_str)

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

        # 工程指标表
        if "engineering_metrics" in results:
            df = results["engineering_metrics"]

            lines.append("% 工程指标对比表")
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            lines.append("\\caption{工程指标对比}")
            lines.append("\\label{tab:engineering_metrics}")

            num_cols = len(df.columns)
            lines.append(f"\\begin{{tabular}}{{l{'c' * num_cols}}}")
            lines.append("\\toprule")

            header = "指标 & " + " & ".join(df.columns) + " \\\\"
            lines.append(header)
            lines.append("\\midrule")

            for idx, row in df.iterrows():
                values = [f"{val:.2f}" if isinstance(val, float) else str(val) for val in row.values]
                row_str = f"{idx} & " + " & ".join(values) + " \\\\"
                lines.append(row_str)

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

        # 写入文件
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"LaTeX 表格已生成: {latex_path}")
        return latex_path

    def _generate_json_report(
        self,
        results: dict[str, Any],
        config: Optional[dict],
        timestamp: str,
    ) -> Path:
        """生成 JSON 格式报告。"""
        json_path = self.output_dir / f"report_{timestamp}.json"

        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "2.0.0",
            },
            "config": config or {},
            "results": {},
        }

        # 转换 DataFrame 为 dict
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                report_data["results"][key] = {
                    "type": "dataframe",
                    "data": value.to_dict(),
                }
            else:
                try:
                    # 尝试序列化
                    json.dumps({key: value})
                    report_data["results"][key] = value
                except (TypeError, ValueError):
                    report_data["results"][key] = str(value)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"JSON 报告已生成: {json_path}")
        return json_path

    def generate_black_swan_report(
        self,
        black_swan_results: dict[str, Any],
        event_name: str,
        event_period: str,
    ) -> Path:
        """
        生成黑天鹅事件专项分析报告。

        Args:
            black_swan_results: 黑天鹅期间回测结果
            event_name: 事件名称
            event_period: 事件时间范围

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"black_swan_{event_name}_{timestamp}.md"

        lines = []

        lines.append(f"# 黑天鹅事件专项分析报告: {event_name}")
        lines.append(f"**分析时间**: {event_period}")
        lines.append(f"**报告生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 事件描述
        lines.append("## 事件概述\n")
        lines.append(f"**事件名称**: {event_name}")
        lines.append(f"**时间范围**: {event_period}\n")

        # 各轨道表现
        lines.append("## 各轨道表现\n")

        if "track_results" in black_swan_results:
            results_df = pd.DataFrame(black_swan_results["track_results"]).T
            lines.append(results_df.to_markdown())
            lines.append("")

        # 风险控制对比
        lines.append("## 风险控制对比\n")

        if "risk_metrics" in black_swan_results:
            risk_df = pd.DataFrame(black_swan_results["risk_metrics"]).T
            lines.append(risk_df.to_markdown())
            lines.append("")

        # LLM vs ML 对比分析
        lines.append("## LLM vs ML 对比分析\n")

        ml_tracks = black_swan_results.get("ml_tracks", [])
        llm_tracks = black_swan_results.get("llm_tracks", [])

        if ml_tracks and llm_tracks:
            ml_avg_dd = results_df.loc[ml_tracks, "max_drawdown"].mean()
            llm_avg_dd = results_df.loc[llm_tracks, "max_drawdown"].mean()

            lines.append(f"- **ML Tracks 平均最大回撤**: {ml_avg_dd:.2%}")
            lines.append(f"- **LLM Tracks 平均最大回撤**: {llm_avg_dd:.2%}")
            lines.append(f"- **回撤差异**: {abs(ml_avg_dd - llm_avg_dd):.2%}")

            if llm_avg_dd < ml_avg_dd:
                lines.append("- **结论**: LLM Tracks 在黑天鹅事件中展现更好的风险控制能力\n")
            else:
                lines.append("- **结论**: ML Tracks 在黑天鹅事件中展现更好的风险控制能力\n")

        # 写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"黑天鹅分析报告已生成: {report_path}")
        return report_path


# 快捷函数
def generate_report(
    results: dict[str, Any],
    config: Optional[dict] = None,
    format: str = "all",
    output_dir: str = "docs/output/reports",
) -> dict[str, Path]:
    """
    快捷函数：生成报告。

    Args:
        results: 实验结果
        config: 实验配置
        format: 输出格式
        output_dir: 输出目录

    Returns:
        生成的报告文件路径
    """
    generator = ReportGenerator(output_dir)
    return generator.generate_full_report(results, config, format)


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("报告生成器示例")
    print("=" * 60)

    # 创建示例数据
    import numpy as np

    np.random.seed(42)

    # 金融指标
    financial_metrics = pd.DataFrame({
        "total_return": [0.15, 0.12, 0.18, 0.10, 0.08],
        "sharpe_ratio": [1.2, 1.0, 1.4, 0.8, 0.7],
        "max_drawdown": [-0.10, -0.15, -0.08, -0.20, -0.12],
        "volatility": [0.15, 0.16, 0.18, 0.14, 0.13],
    }, index=["LR", "LSTM", "LGB", "LLM-Cloud", "LLM-Local"])

    # 工程指标
    engineering_metrics = pd.DataFrame({
        "avg_latency_ms": [2.5, 15.0, 3.0, 800, 1200],
        "throughput_qps": [400, 67, 333, 1.25, 0.83],
        "cost_per_signal": [0.0, 0.0, 0.0, 0.002, 0.001],
    }, index=["LR", "LSTM", "LGB", "LLM-Cloud", "LLM-Local"])

    results = {
        "financial_metrics": financial_metrics,
        "engineering_metrics": engineering_metrics,
        "conclusions": [
            "LGB 模型在收益和夏普比率方面表现最佳",
            "ML Tracks 在延迟和成本方面显著优于 LLM Tracks",
            "LLM Tracks 在风险控制方面展现出潜力",
        ],
    }

    config = {
        "experiment": "Five-Track Comparison",
        "symbol": "CSI300",
        "period": "2020-2024",
        "tracks": ["LR", "LSTM", "LGB", "LLM-Cloud", "LLM-Local"],
    }

    # 生成报告
    generator = ReportGenerator()
    files = generator.generate_full_report(results, config, format="all")

    print("\n生成的报告文件:")
    for fmt, path in files.items():
        print(f"  {fmt}: {path}")

    # 生成黑天鹅报告示例
    black_swan_results = {
        "track_results": {
            "LR": {"max_drawdown": -0.25, "recovery_days": 30},
            "LSTM": {"max_drawdown": -0.30, "recovery_days": 45},
            "LLM-Cloud": {"max_drawdown": -0.18, "recovery_days": 20},
        },
        "ml_tracks": ["LR", "LSTM"],
        "llm_tracks": ["LLM-Cloud"],
    }

    bs_report = generator.generate_black_swan_report(
        black_swan_results,
        event_name="COVID-19 Crash",
        event_period="2020-02-01 ~ 2020-03-31",
    )

    print(f"\n黑天鹅报告: {bs_report}")
