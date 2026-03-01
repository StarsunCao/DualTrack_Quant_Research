# 添加详细交易和持仓记录功能

## 功能说明

为每个轨道保存以下详细信息：
- **交易记录**: 每笔买卖的时间、价格、数量、手续费
- **持仓记录**: 每日持仓市值、现金余额、总资产
- **调仓记录**: 每次调仓的目标仓位和实际执行

---

## 修改步骤

### 1. 修改 `src/execution/bt_engine.py`

#### 1.1 在 `DualTrackStrategy` 类中添加记录功能

```python
class DualTrackStrategy(bt.Strategy):
    """
    双轨策略类。

    新增记录功能：
    - 详细交易记录
    - 每日持仓记录
    - 调仓执行记录
    """

    def __init__(self) -> None:
        """初始化策略。"""
        # ... 原有代码 ...

        # 新增：详细交易记录
        self.trade_records: list[dict] = []  # 交易记录
        self.position_records: list[dict] = []  # 每日持仓记录
        self.rebalance_records: list[dict] = []  # 调仓记录

    def notify_order(self, order: bt.Order) -> None:
        """订单状态通知 - 增强版。"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0)
            record = {
                "date": dt.isoformat(),
                "type": "买入" if order.isbuy() else "卖出",
                "price": order.executed.price,
                "size": order.executed.size,
                "value": order.executed.value,
                "commission": order.executed.comm,
                "symbol": self.datas[0]._name or "CSI300",
            }
            self.trade_records.append(record)

            if order.isbuy():
                self.log(f"买入执行: 价格={order.executed.price:.2f}, 数量={order.executed.size:.0f}, 成本={order.executed.value:.2f}")
            else:
                self.log(f"卖出执行: 价格={order.executed.price:.2f}, 数量={order.executed.size:.0f}, 成本={order.executed.value:.2f}")

            self.trade_count += 1
        # ... 原有代码 ...

    def next(self) -> None:
        """每个 bar 执行的逻辑 - 增强版。"""
        current_date = self.datas[0].datetime.date(0)
        current_value = self.broker.getvalue()

        # 记录每日持仓
        pos = self.getposition(self.datas[0])
        position_record = {
            "date": current_date.isoformat(),
            "symbol": self.datas[0]._name or "CSI300",
            "position_size": pos.size if pos else 0,
            "position_value": pos.size * self.dataclose[0] if pos else 0,
            "cash": self.broker.getcash(),
            "total_value": current_value,
            "close_price": self.dataclose[0],
        }
        self.position_records.append(position_record)

        # ... 原有代码 ...

        # 记录调仓
        if target:
            rebalance_record = {
                "date": current_date.isoformat(),
                "symbol": list(target.keys())[0] if target else "",
                "target_weight": list(target.values())[0] if target else 0,
                "actual_position": pos.size * self.dataclose[0] / current_value if pos and current_value > 0 else 0,
                "portfolio_value": current_value,
            }
            self.rebalance_records.append(rebalance_record)
```

#### 1.2 在 `BacktestEngine.run()` 中传递记录

```python
def run(self) -> BacktestResult:
    """运行回测 - 增强版。"""
    # ... 原有代码 ...

    # 运行回测
    self.cerebro.run()

    # 获取策略实例
    strategy = self.cerebro.runningstrats[0][0]  # 获取第一个策略

    # 构建结果
    result = BacktestResult(
        initial_cash=self.initial_cash,
        final_value=final_value,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_len=max_drawdown_len,
        equity_curve=equity_df,
        trades=pd.DataFrame(strategy.trade_records),  # 交易记录
        analyzers={"trade_analysis": trade_analysis},
    )

    # 附加详细记录
    result.trade_details = pd.DataFrame(strategy.trade_records)
    result.position_details = pd.DataFrame(strategy.position_records)
    result.rebalance_details = pd.DataFrame(strategy.rebalance_records)

    return result
```

### 2. 修改 `main.py` 保存详细记录

```python
def run_backtest(...) -> dict:
    """运行回测并保存详细记录。"""

    # ... 原有代码 ...

    # 运行回测
    track_results = {}
    for track_name, positions in track_positions.items():
        click.echo(f"\n  【轨道: {track_name.upper()}】回测执行...")

        # 执行回测
        result = engine.run()
        track_results[track_name] = result

        # 保存详细记录
        output_dir = Path("docs/output") / track_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存交易记录
        if hasattr(result, 'trade_details') and not result.trade_details.empty:
            result.trade_details.to_csv(
                output_dir / "trades.csv",
                index=False,
                encoding='utf-8'
            )
            click.echo(f"    💾 交易记录: {output_dir}/trades.csv")

        # 2. 保存持仓记录
        if hasattr(result, 'position_details') and not result.position_details.empty:
            result.position_details.to_csv(
                output_dir / "positions.csv",
                index=False,
                encoding='utf-8'
            )
            click.echo(f"    💾 持仓记录: {output_dir}/positions.csv")

        # 3. 保存调仓记录
        if hasattr(result, 'rebalance_details') and not result.rebalance_details.empty:
            result.rebalance_details.to_csv(
                output_dir / "rebalances.csv",
                index=False,
                encoding='utf-8'
            )
            click.echo(f"    💾 调仓记录: {output_dir}/rebalances.csv")

        # 4. 生成交易报告
        generate_trade_report(result, output_dir / "report.txt")

    return track_results

def generate_trade_report(result: BacktestResult, path: Path) -> None:
    """生成交易报告。"""
    lines = [
        "=" * 70,
        "  详细交易报告",
        "=" * 70,
        "",
        "【回测概览】",
        f"  初始资金: {result.initial_cash:,.2f}",
        f"  最终资产: {result.final_value:,.2f}",
        f"  总收益率: {result.total_return:.2%}",
        f"  夏普比率: {result.sharpe_ratio:.4f}",
        "",
    ]

    # 交易统计
    if hasattr(result, 'trade_details') and not result.trade_details.empty:
        trades = result.trade_details
        lines.extend([
            "【交易统计】",
            f"  总交易次数: {len(trades)}",
            f"  买入次数: {len(trades[trades['type'] == '买入'])}",
            f"  卖出次数: {len(trades[trades['type'] == '卖出'])}",
            f"  总手续费: {trades['commission'].sum():.2f}",
            "",
            "【交易明细】",
            trades.to_string(),
            "",
        ])

    # 持仓统计
    if hasattr(result, 'position_details') and not result.position_details.empty:
        positions = result.position_details
        lines.extend([
            "【持仓统计】",
            f"  回测天数: {len(positions)}",
            f"  平均持仓市值: {positions['position_value'].mean():,.2f}",
            f"  平均现金余额: {positions['cash'].mean():,.2f}",
            "",
        ])

    lines.append("=" * 70)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
```

---

## 输出文件结构

运行后将生成以下文件：

```
docs/output/
├── lr/                          # LR轨道
│   ├── trades.csv              # 交易记录
│   ├── positions.csv           # 每日持仓
│   ├── rebalances.csv          # 调仓记录
│   └── report.txt              # 汇总报告
├── lstm/                        # LSTM轨道
│   ├── trades.csv
│   ├── positions.csv
│   ├── rebalances.csv
│   └── report.txt
├── lgb/                         # LightGBM轨道
│   ├── trades.csv
│   ├── positions.csv
│   ├── rebalances.csv
│   └── report.txt
├── llm-cloud/                   # LLM云端轨道
│   └── ...
├── llm-local/                   # LLM本地轨道
│   └── ...
└── figures/
    └── equity_curves_CSI300.png
```

---

## CSV文件格式

### trades.csv（交易记录）

| 字段 | 说明 | 示例 |
|------|------|------|
| date | 交易日期 | 2025-03-15 |
| type | 交易类型 | 买入/卖出 |
| price | 成交价格 | 3950.50 |
| size | 成交数量 | 100 |
| value | 成交金额 | 395050.00 |
| commission | 手续费 | 79.01 |
| symbol | 股票代码 | CSI300 |

### positions.csv（持仓记录）

| 字段 | 说明 | 示例 |
|------|------|------|
| date | 日期 | 2025-03-15 |
| symbol | 股票代码 | CSI300 |
| position_size | 持仓数量 | 100 |
| position_value | 持仓市值 | 395050.00 |
| cash | 现金余额 | 604920.99 |
| total_value | 总资产 | 1000000.00 |
| close_price | 收盘价 | 3950.50 |

### rebalances.csv（调仓记录）

| 字段 | 说明 | 示例 |
|------|------|------|
| date | 调仓日期 | 2025-03-15 |
| symbol | 股票代码 | CSI300 |
| target_weight | 目标权重 | 0.80 |
| actual_position | 实际仓位 | 0.395 |
| portfolio_value | 组合市值 | 1000000.00 |

---

## 使用方法

运行回测后，查看输出：

```bash
# 查看交易记录
cat docs/output/lr/trades.csv

# 查看持仓记录
cat docs/output/lr/positions.csv

# 查看汇总报告
cat docs/output/lr/report.txt
```

使用 Python 分析：

```python
import pandas as pd

# 读取交易记录
trades = pd.read_csv("docs/output/lr/trades.csv")
print(f"总交易次数: {len(trades)}")
print(f"买入次数: {len(trades[trades['type'] == '买入'])}")
print(f"卖出次数: {len(trades[trades['type'] == '卖出'])}")

# 读取持仓记录
positions = pd.read_csv("docs/output/lr/positions.csv")
positions.plot(x='date', y='total_value', title='资产净值曲线')
```

---

## 验证清单

- [ ] 交易记录包含所有买卖操作
- [ ] 持仓记录包含每日数据
- [ ] 调仓记录包含每次调仓
- [ ] CSV文件格式正确
- [ ] 报告文件包含完整统计
- [ ] 五个轨道都有独立输出
