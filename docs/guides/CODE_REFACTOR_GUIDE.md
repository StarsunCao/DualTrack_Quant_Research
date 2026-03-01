# DualTrack Quant Research - 代码审查与修改意见

**版本**: 1.0
**日期**: 2026-03-01
**审查范围**: 全部源代码
**目标**: 修正偏离"对比实验"目标的实现

---

## 🔴 严重问题：概念性偏差

### 问题根源

**错误理解**: 当前代码将 DualTrack 实现为"融合策略"
**正确理解**: DualTrack 应该是"对比框架"，核心是**独立运行 ML Track 和 LLM Track，然后对比分析**

---

## 📋 修改清单

### 1. main.py - 核心流程错误 ⛔⛔⛔

**文件**: `/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/main.py`

#### 问题定位

**Line 204-235**: Phase 4 调用了融合引擎

```python
# ❌ 错误的实现
click.echo("\n[Phase 4/6] 信号融合...")

fusion_engine = SignalFusionEngine(...)
target_positions = fusion_engine.generate_target_positions(
    ml_signals=ml_signals,
    llm_signals=llm_signals,  # ❌ 融合了两个轨道！
    volatility=volatility,
    has_major_news=False,
)
```

**问题**: 融合不是项目的核心目标！这会破坏对比实验的独立性！

#### 修改方案

```python
# ✅ 正确的实现 - 方案1: 分开运行实验
@click.command("run")
@click.option("--experiment", type=click.Choice(["ml", "llm", "both"]), default="both")
@click.pass_context
def run_backtest(ctx, experiment, symbol, start, end, initial_cash, commission, output_dir):
    """
    执行对比实验。

    实验组:
      - ml: 只运行 ML Track (实验A)
      - llm: 只运行 LLM Track (实验B)
      - both: 同时运行两个实验，并对比分析
    """

    results = {}

    # ================================================================
    # Phase 1: 数据获取 (共用)
    # ================================================================
    ohlcv_data, news_data = fetch_data(symbol, start, end)

    # ================================================================
    # 实验A: ML Track
    # ================================================================
    if experiment in ["ml", "both"]:
        click.echo("\n" + "="*70)
        click.echo("  【实验A】ML Track")
        click.echo("="*70)

        # Phase 2: ML 信号生成
        ml_signals = generate_ml_signals(ohlcv_data)

        # 转换为目标仓位 (不融合！)
        ml_positions = signals_to_positions(ml_signals, method="average")

        # Phase 5: ML 回测
        ml_result = run_backtest(
            ohlcv_data=ohlcv_data,
            positions=ml_positions,
            initial_cash=initial_cash,
            commission=commission,
            label="Exp-A_ML_Track"
        )

        results["ML_Track"] = ml_result

        # 保存结果
        save_result(ml_result, f"{output_dir}/exp_a_ml_track.json")

    # ================================================================
    # 实验B: LLM Track
    # ================================================================
    if experiment in ["llm", "both"]:
        click.echo("\n" + "="*70)
        click.echo("  【实验B】LLM Track")
        click.echo("="*70)

        # Phase 3: LLM 信号生成
        llm_signals = generate_llm_signals(news_data)

        # 转换为目标仓位 (不融合！)
        llm_positions = signals_to_positions(llm_signals, method="confidence_weighted")

        # Phase 5: LLM 回测
        llm_result = run_backtest(
            ohlcv_data=ohlcv_data,
            positions=llm_positions,
            initial_cash=initial_cash,
            commission=commission,
            label="Exp-B_LLM_Track"
        )

        results["LLM_Track"] = llm_result

        # 保存结果
        save_result(llm_result, f"{output_dir}/exp_b_llm_track.json")

    # ================================================================
    # Phase 6: 对比分析 (核心！)
    # ================================================================
    if experiment == "both":
        click.echo("\n" + "="*70)
        click.echo("  【对比分析】ML vs LLM")
        click.echo("="*70)

        comparison = compare_experiments(results["ML_Track"], results["LLM_Track"])

        # 打印对比结果
        print_comparison_table(comparison)

        # 保存对比结果
        save_comparison(comparison, f"{output_dir}/comparison_report.json")

    return results


# ================================================================
# 新增辅助函数
# ================================================================

def signals_to_positions(signals: pd.DataFrame, method: str = "average") -> dict:
    """
    将信号转换为目标仓位（不融合，只转换格式）。

    Args:
        signals: 信号DataFrame（ML或LLM）
        method: 转换方法
            - "average": 平均多个模型的信号（ML Track）
            - "confidence_weighted": 置信度加权（LLM Track）

    Returns:
        目标仓位字典 {datetime: {symbol: weight}}
    """
    positions = {}

    if "signal_strength_0_to_1" in signals.columns:
        # ML 信号：0-1 转换为 -1 到 1
        if method == "average":
            grouped = signals.groupby("timestamp")
            for timestamp, group in grouped:
                avg_signal = group["signal_strength_0_to_1"].mean()
                weight = (avg_signal - 0.5) * 2  # 0-1 → -1到1
                symbol = group["symbol"].iloc[0]
                positions[timestamp] = {symbol: weight}

    elif "llm_signal" in signals.columns:
        # LLM 信号：buy/sell/hold 转换为权重
        signal_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}

        if method == "confidence_weighted":
            for _, row in signals.iterrows():
                timestamp = row.get("timestamp", datetime.now())
                symbol = row["symbol"]
                signal = signal_map.get(row["llm_signal"], 0.0)
                confidence = row.get("confidence", 0.5)
                weight = signal * confidence
                positions[timestamp] = {symbol: weight}

    return positions


def compare_experiments(ml_result, llm_result) -> dict:
    """
    对比两个实验的结果。

    Returns:
        对比分析结果
    """
    from src.evaluation.metrics_calculator import MultiStrategyComparator

    comparator = MultiStrategyComparator()
    comparator.add_result("ML_Track", ml_result)
    comparator.add_result("LLM_Track", llm_result)

    financial = comparator.compare_financial_metrics()
    engineering = comparator.compare_engineering_metrics()

    # 核心问题分析
    analysis = {
        "sharpe_winner": "ML" if ml_result.financial_metrics.sharpe_ratio > llm_result.financial_metrics.sharpe_ratio else "LLM",
        "sharpe_diff": abs(ml_result.financial_metrics.sharpe_ratio - llm_result.financial_metrics.sharpe_ratio),
        "drawdown_winner": "ML" if ml_result.financial_metrics.max_drawdown < llm_result.financial_metrics.max_drawdown else "LLM",
        "cost_comparison": {
            "ml_cost": ml_result.engineering_metrics.total_cost_usd,
            "llm_cost": llm_result.engineering_metrics.total_cost_usd,
        },
        "financial_metrics": financial.to_dict(),
        "engineering_metrics": engineering.to_dict(),
    }

    return analysis


def print_comparison_table(comparison: dict):
    """打印对比结果表格。"""
    print("\n" + "="*70)
    print("  核心问题回答")
    print("="*70)

    print("\n【RQ1】谁的收益更高？")
    print(f"  ML Track Sharpe:  {comparison['financial_metrics']['ML_Track']['Sharpe Ratio']}")
    print(f"  LLM Track Sharpe: {comparison['financial_metrics']['LLM_Track']['Sharpe Ratio']}")
    print(f"  ✅ Winner: {comparison['sharpe_winner']} (差异: {comparison['sharpe_diff']:.4f})")

    print("\n【RQ2】谁更稳健？")
    print(f"  ML Track MaxDD:  {comparison['financial_metrics']['ML_Track']['Max Drawdown']}")
    print(f"  LLM Track MaxDD: {comparison['financial_metrics']['LLM_Track']['Max Drawdown']}")
    print(f"  ✅ Winner: {comparison['drawdown_winner']}")

    print("\n【RQ3】成本对比")
    print(f"  ML Cost:  ${comparison['cost_comparison']['ml_cost']:.4f}")
    print(f"  LLM Cost: ${comparison['cost_comparison']['llm_cost']:.4f}")
    print("="*70)
```

---

### 2. src/orchestrator/fusion_engine.py - 定位错误 ⚠️⚠️

**文件**: `/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/src/orchestrator/fusion_engine.py`

#### 问题定位

整个文件实现了"融合引擎"，但**融合不是核心目标**！

**Line 1-5**:
```python
"""
信号融合引擎模块。

实现 ML Track 和 LLM Track 的信号融合，根据市场状态和新闻事件动态调整权重。
"""
```

#### 修改方案

**方案1: 保留但重命名并标注为"可选探索"**

```python
"""
信号融合引擎模块（可选探索）。

⚠️ 注意：此模块不是项目的核心目标！

项目核心：对比 ML Track 和 LLM Track，回答"谁更好"。
此模块提供的融合功能仅作为附加研究，不应在主实验中使用。

适用场景：
  - 探索性研究：融合是否能带来额外收益？
  - 实际应用：如果实验证明两个轨道互补，可考虑融合。

使用建议：
  - 主实验中禁用此模块
  - 在 Exp-C（可选实验）中探索融合效果
"""

# 文件其余部分保持不变，但在文档中明确标注
```

**方案2: 重构为"信号转换器"（推荐）**

```python
"""
信号转换器模块。

将 ML Track 或 LLM Track 的信号转换为目标仓位。
不进行融合，只进行格式转换。

核心原则：
  - ML Track 和 LLM Track 独立运行
  - 每个轨道独立生成目标仓位
  - 对比两个轨道的回测结果
"""

class SignalConverter:
    """
    信号转换器。

    将 ML 或 LLM 信号转换为目标仓位格式。
    """

    @staticmethod
    def ml_signals_to_positions(ml_signals: pd.DataFrame) -> dict:
        """
        将 ML 信号转换为目标仓位。

        Args:
            ml_signals: ML Track 信号 DataFrame

        Returns:
            目标仓位字典 {datetime: {symbol: weight}}
        """
        # 实现同上面的 signals_to_positions 函数
        pass

    @staticmethod
    def llm_signals_to_positions(llm_signals: pd.DataFrame) -> dict:
        """
        将 LLM 信号转换为目标仓位。

        Args:
            llm_signals: LLM Track 信号 DataFrame

        Returns:
            目标仓位字典 {datetime: {symbol: weight}}
        """
        # 实现同上面的 signals_to_positions 函数
        pass


# 可选：保留融合引擎，但重命名
class SignalFusionEngine:
    """
    信号融合引擎（可选探索）。

    ⚠️ 不是核心实验的一部分！
    仅用于 Exp-C 探索融合策略的效果。
    """
    pass
```

---

### 3. 测试文件 test_orchestrator.py - 测试目标错误 ⚠️

**文件**: `/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/tests/test_orchestrator.py`

#### 问题定位

当前测试验证了"融合功能"，但**应该测试对比分析功能**！

#### 修改方案

```python
"""
Orchestrator 模块测试。

验证信号转换和对比分析功能。
"""

def test_signal_conversion():
    """测试信号转换功能。"""
    # 测试 ML 信号转换
    ml_signals = create_mock_ml_signals()
    ml_positions = SignalConverter.ml_signals_to_positions(ml_signals)

    # 验证格式
    assert isinstance(ml_positions, dict)
    assert all(isinstance(v, dict) for v in ml_positions.values())

    # 测试 LLM 信号转换
    llm_signals = create_mock_llm_signals()
    llm_positions = SignalConverter.llm_signals_to_positions(llm_signals)

    # 验证格式
    assert isinstance(llm_positions, dict)
    pass


def test_experiment_comparison():
    """测试实验对比功能。"""
    # 创建两个模拟实验结果
    ml_result = create_mock_result(sharpe=1.2, max_dd=0.15)
    llm_result = create_mock_result(sharpe=0.9, max_dd=0.25)

    # 对比分析
    comparison = compare_experiments(ml_result, llm_result)

    # 验证对比结果
    assert comparison["sharpe_winner"] == "ML"
    assert comparison["drawdown_winner"] == "ML"
    assert comparison["sharpe_diff"] == 0.3


def test_no_fusion_in_main_experiment():
    """验证主实验中没有融合。"""
    # 模拟运行主实验
    results = run_experiment(mode="both")

    # 验证：应该有两个独立的结果
    assert "ML_Track" in results
    assert "LLM_Track" in results

    # 验证：不应该有"Fused"结果
    assert "Fused_Track" not in results
```

---

### 4. 可视化模块 - 图表标题错误 ⚠️

**文件**: `/Users/caoxinyang/PycharmProjects/DualTrack_Quant_Research/src/evaluation/visualizer.py`

#### 问题定位

**Line 54**: 图表标题暗示"融合"

```python
def plot_equity_curves(equity_curves, title="Strategy Comparison: Equity Curves"):
    """
    绘制多策略资金曲线对比图。

    默认绘制三条曲线：ML纯净版、LLM纯净版、双轨融合版
    """
```

#### 修改方案

```python
def plot_equity_curves(equity_curves, title="ML Track vs LLM Track: Equity Curves"):
    """
    绘制 ML Track 和 LLM Track 的资金曲线对比图。

    Args:
        equity_curves: 字典 {策略名: 资金曲线DataFrame}
            - 必须包含 "ML_Track" 和 "LLM_Track"
            - 可选包含 "Fused_Track"（探索性研究）

    Example:
        >>> curves = {
        ...     "ML_Track": ml_equity_df,
        ...     "LLM_Track": llm_equity_df,
        ... }
        >>> plot_equity_curves(curves)
    """
    # 验证输入
    if "ML_Track" not in equity_curves or "LLM_Track" not in equity_curves:
        raise ValueError("必须包含 ML_Track 和 LLM_Track 的资金曲线")

    # 绘制两条主要曲线
    plt.plot(equity_curves["ML_Track"], label="ML Track", linewidth=2)
    plt.plot(equity_curves["LLM_Track"], label="LLM Track", linewidth=2)

    # 可选：绘制融合曲线（如果有）
    if "Fused_Track" in equity_curves:
        plt.plot(equity_curves["Fused_Track"], label="Fused Track (Exploratory)",
                 linewidth=1, linestyle="--", alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
```

---

### 5. 文档字符串和注释 - 概念混淆 ⚠️

#### 全局修改

在所有文件的文档字符串中，修正对 DualTrack 的描述：

**错误示例**:
```python
"""
DualTrack 融合策略回测框架。
核心：融合 ML 和 LLM 信号。
"""
```

**正确示例**:
```python
"""
DualTrack 对比实验框架。
核心：对比 ML Track 和 LLM Track，回答"谁更好"。
"""
```

**修改位置**:
- `main.py` (Line 1-14)
- `src/orchestrator/fusion_engine.py` (Line 1-5)
- `README.md` (项目描述部分)
- `CLAUDE.md` (项目描述部分)

---

## 📝 新增文件建议

### 1. 新增 `src/orchestrator/comparator.py`

```python
"""
实验对比分析模块。

核心功能：对比 ML Track 和 LLM Track 的实验结果。
"""

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from src.evaluation.metrics_calculator import EvaluationResult


@dataclass
class ComparisonResult:
    """对比分析结果。"""

    # 核心问题回答
    sharpe_winner: str  # "ML" 或 "LLM"
    drawdown_winner: str
    return_winner: str

    # 详细指标对比
    financial_comparison: pd.DataFrame
    engineering_comparison: pd.DataFrame

    # 场景分析
    scenario_analysis: Dict[str, Any]

    # 结论
    conclusion: str


def compare_experiments(
    ml_result: EvaluationResult,
    llm_result: EvaluationResult,
) -> ComparisonResult:
    """
    对比 ML Track 和 LLM Track 的实验结果。

    Args:
        ml_result: ML Track 实验结果
        llm_result: LLM Track 实验结果

    Returns:
        对比分析结果
    """
    # 金融指标对比
    financial = _compare_financial_metrics(ml_result, llm_result)

    # 工程指标对比
    engineering = _compare_engineering_metrics(ml_result, llm_result)

    # 回答核心问题
    sharpe_winner = "ML" if ml_result.financial_metrics.sharpe_ratio > llm_result.financial_metrics.sharpe_ratio else "LLM"
    drawdown_winner = "ML" if ml_result.financial_metrics.max_drawdown < llm_result.financial_metrics.max_drawdown else "LLM"
    return_winner = "ML" if ml_result.financial_metrics.total_return > llm_result.financial_metrics.total_return else "LLM"

    # 生成结论
    conclusion = _generate_conclusion(
        sharpe_winner, drawdown_winner, return_winner,
        financial, engineering
    )

    return ComparisonResult(
        sharpe_winner=sharpe_winner,
        drawdown_winner=drawdown_winner,
        return_winner=return_winner,
        financial_comparison=financial,
        engineering_comparison=engineering,
        scenario_analysis={},
        conclusion=conclusion,
    )


def _generate_conclusion(sharpe_winner, drawdown_winner, return_winner, financial, engineering):
    """生成对比结论。"""
    lines = [
        "="*70,
        "  DualTrack 对比实验结论",
        "="*70,
        "",
        "【核心问题回答】",
        "",
        "Q1: 谁的收益更高？",
        f"  ✅ {sharpe_winner} Track 的夏普比率更高",
        "",
        "Q2: 谁更稳健？",
        f"  ✅ {drawdown_winner} Track 的最大回撤更小",
        "",
        "Q3: 成本效益比？",
        "  ✅ ML Track 成本接近于零",
        "  ✅ LLM Track 需要考虑 API 成本",
        "",
        "【实践建议】",
    ]

    if sharpe_winner == "ML" and drawdown_winner == "ML":
        lines.append("  → 推荐使用 ML Track（收益高且稳健）")
    elif sharpe_winner == "LLM" and drawdown_winner == "LLM":
        lines.append("  → 推荐使用 LLM Track（收益高且稳健）")
    else:
        lines.append("  → 根据具体场景选择：")
        lines.append("    - 追求收益：选择 Sharpe Winner")
        lines.append("    - 控制风险：选择 Drawdown Winner")

    lines.append("="*70)

    return "\n".join(lines)
```

---

## ✅ 修改优先级

| 优先级 | 文件 | 问题严重性 | 修改工作量 |
|--------|------|------------|------------|
| **P0** | `main.py` | ⛔⛔⛔ 核心流程错误 | 中等（重构流程） |
| **P0** | `fusion_engine.py` | ⚠️⚠️ 定位错误 | 低（重命名+文档） |
| **P1** | `test_orchestrator.py` | ⚠️ 测试目标错误 | 中等（新增测试） |
| **P1** | `visualizer.py` | ⚠️ 图表标题误导 | 低（修改文字） |
| **P2** | 文档字符串 | ⚠️ 概念混淆 | 低（批量替换） |

---

## 🎯 修改后的项目流程

```
正确的实验流程:

1. 数据准备 (共用)
   ├── 获取 OHLCV 数据
   └── 获取新闻数据

2. 【独立运行】实验A: ML Track
   ├── 特征工程
   ├── 模型训练 (LR/LSTM/LightGBM)
   ├── 信号生成
   ├── 转换为目标仓位 (不融合!)
   └── 独立回测
       → 结果A (Sharpe, MaxDD, etc.)

3. 【独立运行】实验B: LLM Track
   ├── Prompt 构建
   ├── LLM 推理 (Ollama/DeepSeek)
   ├── 信号生成
   ├── 转换为目标仓位 (不融合!)
   └── 独立回测
       → 结果B (Sharpe, MaxDD, etc.)

4. 【核心】对比分析 (A vs B)
   ├── 金融指标对比
   ├── 工程指标对比
   ├── 场景分析
   └── 回答核心问题:
       - Q1: 谁的收益更高？
       - Q2: 谁更稳健？
       - Q3: 成本效益比如何？

5. 论文撰写
   └── 基于对比结果，回答"谁更好"
```

---

## 📋 修改验证清单

修改完成后，请验证以下内容：

### 功能验证
- [ ] `main.py run --experiment ml` 只运行 ML Track
- [ ] `main.py run --experiment llm` 只运行 LLM Track
- [ ] `main.py run --experiment both` 运行两个实验并对比
- [ ] 对比结果中明确显示"谁更好"

### 文档验证
- [ ] 所有文档中不再出现"DualTrack 融合策略优于单一策略"的表述
- [ ] 所有文档明确说明"DualTrack 是对比框架"
- [ ] 融合引擎被标记为"可选探索"

### 测试验证
- [ ] 测试验证了"独立运行"两个轨道
- [ ] 测试验证了"对比分析"功能
- [ ] 测试明确检查"不应该融合"

### 结果验证
- [ ] 生成的图表标题为"ML vs LLM"
- [ ] 对比报告回答了核心问题
- [ ] 没有"Fused Track 优于 ML"的错误结论

---

## 🚀 实施步骤

### Step 1: 备份当前代码

```bash
git checkout -b fix/correct-understanding
git add .
git commit -m "backup: before fixing conceptual misunderstanding"
```

### Step 2: 修改 main.py

1. 删除 Phase 4 的融合调用
2. 新增 `signals_to_positions()` 函数
3. 新增 `compare_experiments()` 函数
4. 修改命令行参数，支持 `--experiment` 选项

### Step 3: 修改 fusion_engine.py

1. 在文档字符串中添加"可选探索"标注
2. 新增 `SignalConverter` 类
3. 保留融合功能，但明确标注用途

### Step 4: 新增 comparator.py

创建新的对比分析模块

### Step 5: 更新测试

1. 修改 `test_orchestrator.py`
2. 新增 `test_comparator.py`

### Step 6: 更新文档

1. 修改所有文档字符串
2. 更新 README.md
3. 更新 CLAUDE.md

### Step 7: 验证修改

```bash
# 运行所有测试
python -m pytest tests/

# 运行 ML 实验
python main.py run --experiment ml

# 运行 LLM 实验
python main.py run --experiment llm

# 运行对比实验
python main.py run --experiment both
```

---

**修改完成标志**:

1. ✅ 主流程中不再调用融合引擎
2. ✅ 独立运行 ML Track 和 LLM Track
3. ✅ 生成对比分析报告
4. ✅ 文档中明确说明"对比框架"
5. ✅ 测试通过

---

**审查人**: [待填写]
**批准人**: [待填写]
**实施日期**: [待填写]