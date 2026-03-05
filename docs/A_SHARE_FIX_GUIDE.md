# A股回测修复指南

## 四大关键陷阱修复方案

### 陷阱1：T+0交易制度 ⚠️
**问题**：Backtrader默认允许T+0交易，A股严格禁止。

**修复方案**：
```python
class DualTrackStrategy(bt.Strategy):
    def __init__(self):
        self.buy_bar_index = {}  # 记录买入的bar索引

    def next(self):
        # 在卖出前检查是否满足T+1
        current_bar = len(self)
        if data in self.buy_bar_index:
            buy_bar = self.buy_bar_index[data]
            if current_bar <= buy_bar:
                # 不满足T+1，禁止卖出
                return
```

### 陷阱2：零股交易 ⚠️
**问题**：Backtrader默认计算股数，不保证整手。

**修复方案**：
```python
# 使用自定义Sizer
from src.execution.a_share_sizer import AShareSizer
cerebro.addsizer(AShareSizer)
```

### 陷阱3：手续费最低消费 ⚠️
**问题**：A股佣金最低5元，未实现。

**修复方案**：
```python
# 使用自定义Commission
from src.execution.a_share_sizer import AShareCommission
cerebro.broker.addcommissioninfo(AShareCommission())
```

### 陷阱4：成交价格逻辑 ✅
**现状**：已使用T日开盘价成交（正确）。

**验证方法**：
```python
# 检查成交价是否接近开盘价
if abs(executed_price - open_price) < 1:
    print("✅ 使用T日开盘价成交")
```

---

## 实施步骤

1. **修改 `src/execution/bt_engine.py`**
   - 导入 `AShareSizer` 和 `AShareCommission`
   - 在 `BacktestEngine.__init__()` 中添加A股规则
   - 在 `DualTrackStrategy` 中添加T+1检查

2. **添加调仓死区**
   ```python
   # 在DualTrackStrategy.next()中
   if abs(new_weight - current_weight) < 0.05:  # 5%死区
       return  # 不调仓
   ```

3. **重新运行回测验证**
   ```bash
   uv run python main.py run --track lr --symbol CSI300
   ```

---

## 预期改进

- **零股交易**：0笔（全部整手）
- **手续费**：增加约20-30%（最低消费）
- **T+0交易**：0笔（严格禁止）
- **调仓频率**：降低约30-40%（死区过滤）

---

## 论文答辩要点

**明确说明**：
1. "本系统严格遵循A股T+1制度，通过bar索引锁机制防止当日买卖。"
2. "买入操作强制取整至100股倍数，符合A股整手交易规则。"
3. "佣金设置最低5元消费，印花税仅卖出收取0.1%，符合A股实际费用结构。"
4. "引入5%调仓死区，避免频繁小额交易的手续费磨损。"
5. "所有信号基于T-1日数据产生，在T日以开盘价执行。"

---

*Created: 2026-03-04*
*Status: 修复方案已准备，待实施*