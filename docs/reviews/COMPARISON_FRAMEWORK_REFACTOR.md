# DualTrack 对比框架重构记录

**日期**: 2026-03-01
**依据**: `docs/CODE_REFACTOR_GUIDE.md`
**核心目标**: 将"融合策略"修正为"对比框架"

---

## 问题诊断

### 原始误解
- ❌ 代码将 DualTrack 实现为"融合策略"
- ❌ Phase 4 调用融合引擎，混合了 ML 和 LLM 信号
- ❌ 破坏了对比实验的独立性

### 修正理解
- ✅ DualTrack 是"对比框架"
- ✅ 独立运行 ML Track 和 LLM Track
- ✅ 对比分析两个轨道的回测结果
- ✅ 回答核心问题："谁更好？"

---

## 实施修改

### 1. 新增对比分析模块 (src/orchestrator/comparator.py)

**创建文件**: 全新的对比分析模块

**核心组件**:
- `ComparisonResult` 数据类：存储对比结果
- `compare_experiments()`: 对比 ML 和 LLM 实验结果
- `print_comparison_table()`: 打印对比报告

**回答的核心问题**:
- Q1: 谁的收益更高？（Sharpe Ratio 对比）
- Q2: 谁更稳健？（Max Drawdown 对比）
- Q3: 成本效益比如何？（Cost-per-Alpha 对比）

### 2. 修改融合引擎 (src/orchestrator/fusion_engine.py)

**文档字符串更新**:
```python
"""
信号融合引擎模块（可选探索）。

⚠️ 注意：此模块不是项目的核心目标！

项目核心：对比 ML Track 和 LLM Track，回答"谁更好"。
此模块提供的融合功能仅作为附加研究，不应在主实验中使用。
"""
```

**新增 SignalConverter 类**:
- `ml_signals_to_positions()`: 将 ML 信号转换为目标仓位
- `llm_signals_to_positions()`: 将 LLM 信号转换为目标仓位
- 不进行融合，只进行格式转换

### 3. 重构主入口 (main.py)

**新增 `--experiment` 参数**:
```python
@click.option("--experiment", "-e", type=click.Choice(["ml", "llm", "both"]), default="both")
```

**修改实验流程**:
```
正确的实验流程:
1. 数据准备 (共用)
2. 【独立运行】实验A: ML Track
   - ML 信号生成 → 转换为目标仓位 → 独立回测 → 结果A
3. 【独立运行】实验B: LLM Track
   - LLM 信号生成 → 转换为目标仓位 → 独立回测 → 结果B
4. 【核心】对比分析 (A vs B)
   - 金融指标对比
   - 工程指标对比
   - 回答核心问题
```

**移除的内容**:
- ❌ 删除 Phase 4 的融合调用
- ❌ 删除 `target_positions` 融合逻辑

**新增的内容**:
- ✅ `ml_positions`: ML Track 独立目标仓位
- ✅ `llm_positions`: LLM Track 独立目标仓位
- ✅ 独立回测执行（ML Track 和 LLM Track 分别运行）
- ✅ 对比分析报告生成

### 4. 更新可视化模块 (src/evaluation/visualizer.py)

**文档更新**:
- 移除"双轨融合版"默认显示
- 明确说明"对比框架"

**图表逻辑更新**:
- ML/LLM Track：实线（主要）
- Fusion Track：虚线（可选探索）

**颜色映射更新**:
```python
color_map = {
    "ML_Track": COLORS["ml"],
    "LLM_Track": COLORS["llm"],
    "Fused_Track": COLORS["fusion"],  # 可选
}
```

### 5. 更新测试文件 (tests/test_orchestrator.py)

**新增验证点**:

**验证点 7: 信号转换（对比框架）**
- 测试 ML 信号转换
- 测试 LLM 信号转换
- 验证无融合操作

**验证点 8: 实验对比（对比框架核心）**
- 测试对比分析功能
- 验证核心问题回答（Q1/Q2/Q3）

**验证点 9: 无融合约束（关键约束）**
- 验证 ML_Track 独立存在
- 验证 LLM_Track 独立存在
- 验证 Fused_Track 不存在

---

## 验证结果

### 所有测试通过
```bash
$ python tests/test_orchestrator.py

======================================================================
  🚀 双轨编排器核心验证：对比框架测试
======================================================================

✅ 验证点 1: 时间序列对齐 - 通过
✅ 验证点 2: 正常模式验证 - 通过
✅ 验证点 3: 黑天鹅一票否决 - 通过
✅ 验证点 4: 前后对比 - 通过
✅ 验证点 5: 延迟记录 - 通过
✅ 验证点 6: 输出结构 - 通过
✅ 验证点 7: 信号转换（对比框架）- 通过
✅ 验证点 8: 实验对比（对比框架核心）- 通过
✅ 验证点 9: 无融合约束（关键约束）- 通过

🎯 结论: 对比框架验证通过！
```

### 模块导入测试
```bash
$ python -c "from src.orchestrator.comparator import compare_experiments; print('OK')"
OK

$ python -c "from src.orchestrator.fusion_engine import SignalConverter; print('OK')"
OK
```

---

## CLI 使用示例

### 运行对比实验（默认）
```bash
python main.py run --experiment both --symbol CSI300
```
输出：
- ML Track 独立回测结果
- LLM Track 独立回测结果
- 对比分析报告（谁更好）

### 仅运行 ML Track
```bash
python main.py run --experiment ml --symbol CSI300
```

### 仅运行 LLM Track
```bash
python main.py run --experiment llm --symbol CSI300
```

---

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `src/orchestrator/comparator.py` | 新增 | 对比分析模块 |
| `src/orchestrator/fusion_engine.py` | 修改 | 添加 SignalConverter，标注"可选探索" |
| `main.py` | 重构 | 支持 --experiment 参数，独立运行两个轨道 |
| `src/evaluation/visualizer.py` | 修改 | 更新图表标题和融合曲线显示逻辑 |
| `tests/test_orchestrator.py` | 修改 | 添加验证点 7-9 |

---

## 核心原则

### 对比框架（正确）
1. ML Track 和 LLM Track **独立运行**
2. 各自生成目标仓位（**不融合**）
3. 分别执行回测
4. 对比分析结果，回答"谁更好"

### 融合策略（错误）
1. ❌ 混合 ML 和 LLM 信号
2. ❌ 生成融合后的目标仓位
3. ❌ 只执行一次回测（融合策略）
4. ❌ 对比"融合策略 vs 单一策略"

---

## 后续建议

1. **论文撰写**: 基于对比结果回答"谁更好"，而非"融合优于单一"
2. **融合研究**: 如需探索融合效果，使用 `Fused_Track` 作为可选实验
3. **实验设计**: 确保所有实验遵循"独立运行 + 对比分析"原则

---

**重构完成日期**: 2026-03-01
**重构人员**: Claude Code
**审查状态**: 待审查
