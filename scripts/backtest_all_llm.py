#!/usr/bin/env python
"""批量回测所有已生成的 LLM Agent 缓存。"""

import sys
import json
import glob
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from scripts.backtest_agent_vs_ml import load_agent_cache, backtest_track


def run_backtest(cache_path: str, symbol: str, ohlcv_data: pd.DataFrame,
                 commission: float, allow_short: bool) -> dict:
    """回测单个缓存文件。"""
    name = Path(cache_path).stem.replace("llm_cache_", "").replace("_agent", "")

    print(f"\n{'='*60}")
    print(f"  回测: {name}")
    print(f"{'='*60}")

    signals = load_agent_cache(cache_path, symbol)
    print(f"  信号数: {len(signals)}")

    errs = (signals['signal'] == 'hold') & (signals['confidence'] == 0.0)
    print(f"  异常信号(conf=0): {errs.sum()}")

    result = backtest_track(name, signals, ohlcv_data, symbol,
                            commission=commission, allow_short=allow_short)
    if result:
        print(f"  最终资产: {result.final_value:,.2f}")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.4f}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  日换手:   {getattr(result, 'turnover', 0):.4f}")
        return {
            "name": name,
            "final_value": result.final_value,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "turnover": getattr(result, 'turnover', 0),
        }
    return None


def main():
    print("="*60)
    print("  批量回测所有 LLM Agent 缓存")
    print("="*60)

    all_results = []

    for symbol, start, end, commission, allow_short, ohlcv_file in [
        ("CSI300", "2020-01-02", "2024-12-31", 0.0012, False, "data/raw/real_csi300_5y.csv"),
        ("QQQ", "2018-01-02", "2020-07-22", 0.0005, True, "data/raw/real_qqq_5y.csv"),
    ]:
        print(f"\n{'='*60}")
        print(f"  市场: {symbol} ({start} ~ {end})")
        print(f"{'='*60}")

        # 加载 OHLCV
        ohlcv = pd.read_csv(ohlcv_file, parse_dates=["date"])
        ohlcv.set_index("date", inplace=True)
        ohlcv_data = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]
        print(f"  OHLCV: {len(ohlcv_data)} 条")

        # 查找该市场的所有 agent 缓存
        pattern = f"data/llm_cache/llm_cache_{symbol}_*_agent.jsonl"
        cache_files = sorted(glob.glob(pattern))
        print(f"  找到 {len(cache_files)} 个缓存文件")

        for cf in cache_files:
            result = run_backtest(cf, symbol, ohlcv_data, commission, allow_short)
            if result:
                result["symbol"] = symbol
                all_results.append(result)

    # 汇总表格
    if all_results:
        print(f"\n{'='*80}")
        print(f"  全部回测结果汇总")
        print(f"{'='*80}")
        df = pd.DataFrame(all_results)
        # 按市场分组显示
        for sym in ["CSI300", "QQQ"]:
            sym_df = df[df["symbol"] == sym].sort_values("sharpe_ratio", ascending=False)
            if sym_df.empty:
                continue
            print(f"\n  【{sym}】")
            print(f"  {'模型':<35} {'收益率':>10} {'夏普':>8} {'回撤':>8} {'换手':>8}")
            print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
            for _, row in sym_df.iterrows():
                print(f"  {row['name']:<35} {row['total_return']:>9.2%} "
                      f"{row['sharpe_ratio']:>7.4f} {row['max_drawdown']:>7.2%} "
                      f"{row['turnover']:>7.4f}")

        # 保存 CSV
        out_path = Path("docs/output/llm_backtest_summary.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
