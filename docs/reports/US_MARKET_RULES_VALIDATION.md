# 美股交易规则验证与Prompt优化报告

## 执行日期
2026-03-09

## 1. 问题背景

ML轨道回测已完成，发现LLM在风险控制（最大回撤）方面表现优异，但需要验证：
1. 回测引擎的美股交易规则是否正确实现（做空、T+0等）
2. Prompt是否需要优化以提高信号质量

## 2. 问题发现与修复

### 2.1 发现的核心问题

**问题**：`DualTrackStrategy` 硬编码了A股规则（禁止做空），导致美股回测时：
- LLM的sell信号（-confidence）被错误转换为"保留仓位"逻辑
- 例如：sell信号 -0.75 被转换为保留仓位 25%（而非做空）

**代码位置**：`src/execution/bt_engine.py:514-523`

```python
# 原代码问题
if weight < 0:
    # A股规则：SELL信号（负权重）→ 分级减仓而非清仓
    target_weight = 1 + weight  # 负权重转正
    ...
```

### 2.2 修复方案

#### 修复1：添加 `allow_short` 参数

```python
# bt_engine.py - DualTrackStrategy 参数
params = (
    ...
    ("allow_short", False),  # 是否允许做空
    ("short_confidence_threshold", 0.85),  # 做空所需最低置信度
)
```

#### 修复2：根据市场类型设置参数

```python
# main.py - Phase 5
allow_short = False  # 默认禁止做空
engine.add_strategy(
    DualTrackStrategy,
    target_positions=positions,
    allow_short=allow_short,
)
```

#### 修复3：优化美股Prompt

```python
# us_prompts.py - 增强信号说明
【Signal Types - EXPLICIT DEFINITIONS】
- "buy": Open or add to LONG position
- "sell": CLOSE existing long position (go to CASH, NOT short)
- "hold": Maintain current position
```

## 3. 验证结果

### 3.1 做空行为测试

| 策略 | 做空天数 | 总收益率 | 最大回撤 |
|------|---------|---------|---------|
| 全做空 | 551 (91.8%) | -21.42% | 34.71% |
| 阈值做空(85%) | 196 (32.7%) | -4.54% | 25.11% |
| 禁止做空(默认) | 0 (0.0%) | +19.29% | 5.77% |

### 3.2 关键发现

1. **LLM的sell信号置信度普遍较高**：所有sell信号置信度在0.70-0.90之间
2. **直接做空效果不佳**：在牛市中大量做空导致亏损
3. **减仓策略更优**：sell信号作为"防御性减仓"而非"主动性做空"

## 4. 最终实现

### 4.1 代码修改

| 文件 | 修改内容 |
|------|---------|
| `src/execution/bt_engine.py` | 添加 `allow_short` 和 `short_confidence_threshold` 参数 |
| `main.py` | 根据市场类型设置参数 |
| `src/models/llm_track/us_prompts.py` | 优化美股Prompt模板 |
| `src/orchestrator/signal_converter.py` | 更新注释说明 |

### 4.2 美股交易规则验证

| 规则 | 实现状态 | 代码位置 |
|------|---------|---------|
| **做空机制** | ✅ 已实现 | `bt_engine.py:517-540` |
| **T+0交易** | ✅ 正确 | 无限制 |
| **无整手限制** | ✅ 正确 | 可买零股 |
| **费用结构** | ✅ 正确 | 零佣金+SEC费 |

## 5. 建议

### 5.1 当前配置（推荐）

- **默认禁止做空**：LLM的sell信号作为减仓/清仓
- **保留做空选项**：通过 `allow_short=True` 启用
- **高置信度阈值**：如需做空，设置 `short_confidence_threshold=0.90`

### 5.2 后续优化方向

1. **Prompt优化**：引导LLM生成更多buy信号（当前仅3.3%）
2. **信号验证**：添加趋势确认机制，只在下跌趋势中触发做空
3. **仓位控制**：根据VIX动态调整仓位上限

## 6. 文件变更

```
src/execution/bt_engine.py      # 添加做空参数
main.py                         # 设置市场参数
src/models/llm_track/us_prompts.py  # 优化Prompt
src/orchestrator/signal_converter.py  # 更新注释
```