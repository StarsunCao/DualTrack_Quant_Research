# DualTrack Quant Research - 正确理解与实验规划

**版本**: 2.0（修正版）
**日期**: 2026-03-01
**状态**: Phase 1-7 已完成，进入对比实验阶段

---

## ⚠️ 重要修正：项目核心目标

### ❌ 之前的错误理解

- DualTrack 是一个"融合策略"
- 实验目标是证明"融合策略优于单一策略"

### ✅ 正确的理解

**DualTrack 是一个对比框架**，核心目标是：

> **严格对比传统机器学习（Fitting）与大语言模型智能体（Semantic Reasoning）在量化交易中的 ROI 和鲁棒性，特别是在黑天鹅事件下的表现。**

**DualTrack** 的含义：
- **Dual**: 两条轨道（ML Track + LLM Track）
- **Track**: 实验轨道，每个轨道是一个独立的方案
- **目的**: 公平对比两种方案，回答"谁更好"的问题

---

## 🎯 核心研究问题

| 研究问题 | 具体内容 |
|---------|---------|
| **RQ1** | ML vs LLM：谁的收益更高？（ROI对比） |
| **RQ2** | 黑天鹅事件中，谁更稳健？（鲁棒性对比） |
| **RQ3** | LLM 的语义推理能力是否带来超额收益？ |
| **RQ4** | 成本效益比如何？（Cost-per-Alpha对比） |
| **RQ5** | 本地LLM vs 云端LLM：效果与成本对比 |

---

## 📊 正确的实验设计

### 实验组设计

| 实验组 | ML Track | LLM Track | 数据集 | 目的 |
|--------|----------|-----------|--------|------|
| **Exp-A** | ✅ | ❌ | CSI300 (2020-2024) | ML方案基准 |
| **Exp-B** | ❌ | ✅ | CSI300 (2020-2024) | LLM方案基准 |
| **Exp-C** (可选) | ✅ | ✅ | CSI300 (2020-2024) | 组合探索（非核心） |

**重要**：论文的核心是 **对比 Exp-A vs Exp-B**！

### 实验流程

```
┌─────────────────────────────────────────────────────────┐
│                   DualTrack 实验框架                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  数据准备 (共用)                                         │
│  ├── OHLCV数据 (沪深300/QQQ)                            │
│  ├── 新闻数据                                            │
│  └── 时间对齐                                            │
│                                                         │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │   ML Track       │      │   LLM Track      │        │
│  │  (方案A)         │      │  (方案B)         │        │
│  ├──────────────────┤      ├──────────────────┤        │
│  │ 1. 特征工程       │      │ 1. Prompt构建    │        │
│  │ 2. 模型训练       │      │ 2. LLM推理       │        │
│  │  - LR            │      │  - Ollama        │        │
│  │  - LSTM          │      │  - DeepSeek      │        │
│  │  - LightGBM      │      │ 3. 信号生成      │        │
│  │ 3. 信号生成       │      │ 4. 离线缓存      │        │
│  └──────────────────┘      └──────────────────┘        │
│           │                         │                  │
│           │   ✅ 独立运行，分别回测   │                  │
│           │                         │                  │
│           ▼                         ▼                  │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  回测A (Backtest) │      │  回测B (Backtest) │        │
│  └──────────────────┘      └──────────────────┘        │
│           │                         │                  │
│           └─────────┬───────────────┘                  │
│                     ▼                                   │
│           ┌──────────────────┐                         │
│           │   对比分析        │                         │
│           │  A vs B           │                         │
│           ├──────────────────┤                         │
│           │ • 金融指标对比    │                         │
│           │ • 工程指标对比    │                         │
│           │ • 场景分析        │                         │
│           └──────────────────┘                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 当前代码的偏差与修正

### 偏差1: `fusion_engine.py` 的定位错误

**当前实现**：
```python
# src/orchestrator/fusion_engine.py
class SignalFusionEngine:
    """融合 ML 和 LLM 信号"""
    def generate_target_positions(self, ml_signals, llm_signals):
        # 融合两个轨道的信号
        fused_signal = ml_weight * ml_signal + llm_weight * llm_signal
        ...
```

**问题**：
- ❌ 这不是实验的核心目标
- ❌ 融合不是必需的，甚至可能干扰对比分析

**修正方案**：
```python
# 方案1: 保留融合引擎，但标记为"可选探索"
# docs/EXPERIMENT_PLAN.md
## Exp-C (可选): 组合方案探索
## 注：这不是论文的核心，仅作为附加研究

# 方案2: 重构为"对比引擎"
class SignalComparisonEngine:
    """对比 ML 和 LLM 信号"""
    def compare_signals(self, ml_signals, llm_signals, actual_returns):
        """对比两个轨道的信号质量"""
        ml_accuracy = self._evaluate_signals(ml_signals, actual_returns)
        llm_accuracy = self._evaluate_signals(llm_signals, actual_returns)
        return {
            "ml_accuracy": ml_accuracy,
            "llm_accuracy": llm_accuracy,
            "winner": "ML" if ml_accuracy > llm_accuracy else "LLM"
        }
```

### 偏差2: `main.py` 流程错误

**当前流程**：
```python
# main.py (错误)
ml_signals = generate_ml_signals()
llm_signals = generate_llm_signals()
fused_positions = fusion_engine.generate_target_positions(ml_signals, llm_signals)  # ❌ 融合
backtest(fused_positions)
```

**正确流程**：
```python
# main.py (正确)
# 实验A: ML Track
ml_signals = generate_ml_signals()
ml_positions = signals_to_positions(ml_signals)  # 直接转换为目标仓位
ml_result = backtest(ml_positions, label="Exp-A_ML_Track")

# 实验B: LLM Track
llm_signals = generate_llm_signals()
llm_positions = signals_to_positions(llm_signals)
llm_result = backtest(llm_positions, label="Exp-B_LLM_Track")

# 对比分析
compare_results(ml_result, llm_result)
```

---

## 📋 Phase 8-10 正确规划

### Phase 8: 独立实验准备

#### 8.1 数据准备（共用）
- [ ] 获取 2020-2024 年历史数据
- [ ] 生成/收集新闻数据
- [ ] 数据对齐与清洗

#### 8.2 ML Track 实验
- [ ] 特征工程管道
- [ ] 训练 ML 模型（LR, LSTM, LightGBM）
- [ ] 生成信号并保存
- [ ] 独立回测
- [ ] 保存回测结果

#### 8.3 LLM Track 实验
- [ ] 构建 Prompt 模板
- [ ] 构建 LLM 缓存（Ollama/DeepSeek）
- [ ] 生成信号并保存
- [ ] 独立回测
- [ ] 保存回测结果

### Phase 9: 对比实验与分析

#### 9.1 金融指标对比
```python
# scripts/compare_experiments.py
from src.evaluation.metrics_calculator import MultiStrategyComparator

comparator = MultiStrategyComparator()
comparator.add_result("ML_Track", ml_result)
comparator.add_result("LLM_Track", llm_result)

# 生成对比表格
financial_comparison = comparator.compare_financial_metrics()
engineering_comparison = comparator.compare_engineering_metrics()

# 核心问题分析
print("=" * 60)
print("  RQ1: 谁的收益更高？")
print(f"  ML Track Sharpe:  {ml_result.financial_metrics.sharpe_ratio:.4f}")
print(f"  LLM Track Sharpe: {llm_result.financial_metrics.sharpe_ratio:.4f}")
print(f"  Winner: {'ML' if ml_result.financial_metrics.sharpe_ratio > llm_result.financial_metrics.sharpe_ratio else 'LLM'}")
print("=" * 60)
```

#### 9.2 场景分析
- [ ] 正常市场 (2020-2021): ML vs LLM
- [ ] 高波动市场 (2022 俄乌冲突): ML vs LLM
- [ ] 极端行情 (2020 疫情暴跌): ML vs LLM
- [ ] 单边上涨/下跌: ML vs LLM

#### 9.3 成本效益分析
```python
# LLM Track 成本分析
llm_cost_per_alpha = llm_result.engineering_metrics.cost_per_alpha
ml_cost_per_alpha = ml_result.engineering_metrics.cost_per_alpha  # ≈ 0

print(f"LLM Cost-per-Alpha: ${llm_cost_per_alpha:.6f}")
print(f"ML Cost-per-Alpha: ${ml_cost_per_alpha:.6f}")

# 计算"成本调整后收益"
llm_adjusted_return = llm_return - llm_total_cost
ml_adjusted_return = ml_return  # ML成本≈0

# 对比
print(f"成本调整后:")
print(f"  ML:  {ml_adjusted_return:.2%}")
print(f"  LLM: {llm_adjusted_return:.2%}")
```

### Phase 10: 论文撰写

#### 10.1 论文结构
```
1. Introduction
   - 研究背景：ML vs LLM 在量化交易中的应用
   - 研究问题：谁的收益更高？谁更稳健？

2. Methodology
   - ML Track 方法
   - LLM Track 方法
   - 实验设计（对比实验，非融合）

3. Experiments
   - 实验A: ML Track 结果
   - 实验B: LLM Track 结果
   - 对比分析 (A vs B)

4. Results
   - 金融指标对比表格
   - 不同市场环境下的表现对比
   - 成本效益分析

5. Discussion
   - ML 的优势与劣势
   - LLM 的优势与劣势
   - 何时选择 ML？何时选择 LLM？

6. Conclusion
   - 回答研究问题
   - 实践建议
```

#### 10.2 核心结论（预期）
```
论文不应该证明"融合策略更好"，而应该回答：

✅ 正确的结论示例：
  "在正常市场环境下，ML Track 的夏普比率为 1.2，
   LLM Track 为 0.9，ML 表现更优。
   但在黑天鹅事件中，LLM Track 的最大回撤为 -15%，
   ML Track 为 -25%，LLM 展现出更好的风险控制能力。
   考虑成本后，ML Track 的成本效益比更高。"

❌ 错误的结论示例：
  "DualTrack 融合策略的夏普比率为 1.5，
   优于单一的 ML Track (1.2) 和 LLM Track (0.9)，
   证明了融合策略的优越性。"
   （这是偏离研究目标的！）
```

---

## 🛠️ 代码修正建议

### 建议1: 重命名模块

```
当前结构:
src/orchestrator/fusion_engine.py  ❌

修正为:
src/orchestrator/signal_converter.py  ✅
# 功能：将 ML 信号或 LLM 信号转换为目标仓位
# 不做融合，只做格式转换
```

### 建议2: 修改 main.py

```python
# main.py 修正版

@click.command()
@click.option('--experiment', type=click.Choice(['ml', 'llm', 'both']))
def run(experiment):
    """运行实验"""

    if experiment in ['ml', 'both']:
        # 实验A: ML Track
        click.echo("\n" + "="*60)
        click.echo("  实验A: ML Track")
        click.echo("="*60)

        ml_signals = generate_ml_signals()
        ml_result = run_backtest(ml_signals, label="Exp-A_ML_Track")
        save_result(ml_result, "exp_a_ml_track.json")

    if experiment in ['llm', 'both']:
        # 实验B: LLM Track
        click.echo("\n" + "="*60)
        click.echo("  实验B: LLM Track")
        click.echo("="*60)

        llm_signals = generate_llm_signals()
        llm_result = run_backtest(llm_signals, label="Exp-B_LLM_Track")
        save_result(llm_result, "exp_b_llm_track.json")

    if experiment == 'both':
        # 对比分析
        click.echo("\n" + "="*60)
        click.echo("  对比分析: ML vs LLM")
        click.echo("="*60)

        compare_results(ml_result, llm_result)
```

---

## 📊 实验输出文件结构

```
docs/output/
├── experiments/
│   ├── exp_a_ml_track/          # ML Track 实验结果
│   │   ├── signals.parquet      # ML信号
│   │   ├── backtest_result.json # 回测结果
│   │   └── equity_curve.csv     # 资金曲线
│   │
│   └── exp_b_llm_track/         # LLM Track 实验结果
│       ├── signals.parquet      # LLM信号
│       ├── backtest_result.json # 回测结果
│       ├── equity_curve.csv     # 资金曲线
│       └── llm_cache.jsonl      # LLM缓存
│
├── comparison/                   # 对比分析
│   ├── financial_comparison.csv  # 金融指标对比表
│   ├── engineering_comparison.csv # 工程指标对比表
│   ├── scenario_analysis.md      # 场景分析报告
│   └── cost_benefit_analysis.md  # 成本效益分析
│
└── figures/
    ├── ml_vs_llm_equity_curves.png  # 资金曲线对比
    ├── ml_vs_llm_drawdown.png       # 回撤对比
    └── cost_comparison.png          # 成本对比
```

---

## ✅ 验证清单

### 实验设计验证
- [ ] 实验是否专注于"对比"而非"融合"？
- [ ] 论文结论是否回答了"谁更好"的问题？
- [ ] 是否独立运行了 ML Track 和 LLM Track？

### 代码实现验证
- [ ] `main.py` 是否分开运行 ML 和 LLM 实验？
- [ ] 是否有独立的对比分析模块？
- [ ] `fusion_engine.py` 是否被标记为"可选探索"？

---

## 🎯 总结

### 核心修正
1. **DualTrack 是对比框架，不是融合策略**
2. **实验目标：对比 ML vs LLM，谁更好？**
3. **论文核心：独立实验 + 对比分析**
4. **融合引擎：标记为可选探索，非核心目标**

### 下一步行动
1. ✅ 修正规划文档（本文档）
2. 🚧 修改 `main.py` 流程
3. 🚧 调整实验设计
4. 🚧 更新论文写作大纲

---

**文档维护**: 随实验进展更新
**审核状态**: 待用户确认理解是否正确