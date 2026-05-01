#!/usr/bin/env python
"""Find optimal thresholds - A-share and US tested independently."""

import sys, glob, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.signal_converter import SignalConverter
from src.utils.time_utils import aggregate_daily_signals, align_to_trading_days, fill_missing_trading_days
from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

PARAMS = {
    'CSI300': {'path': 'data/raw/real_csi300_5y.csv', 'start': '2020-01-02', 'end': '2024-12-31', 'commission': 0.0012, 'allow_short': False, 'ema': 0.30, 'decay': 0.70},
    'QQQ': {'path': 'data/raw/real_qqq_5y.csv', 'start': '2018-01-02', 'end': '2020-07-22', 'commission': 0.0005, 'allow_short': True, 'ema': 0.50, 'decay': 0.80},
}

def run_one(symbol, wb, ws):
    p = PARAMS[symbol]
    ohlcv = pd.read_csv(p['path'], parse_dates=['date']).set_index('date')
    ohlcv = ohlcv[(ohlcv.index >= p['start']) & (ohlcv.index <= p['end'])]
    results = {}
    for cf in sorted(glob.glob(f'data/llm_cache/llm_cache_{symbol}_*_agent.jsonl')):
        sig = pd.read_json(cf, lines=True)
        sig['timestamp'] = pd.to_datetime(sig['timestamp'])
        daily = aggregate_daily_signals(sig, 'timestamp', 'signal', 'confidence', 'symbol')
        def s2w(s, c):
            s = s.lower().strip()
            if s == 'buy': return 0 if c < 0.5 else (c - 0.5) * 2
            if s == 'short': return 0 if c < 0.5 else -(c - 0.5) * 2
            if s in ('neutral', 'sell'): return 0
            return None
        daily['weight'] = daily.apply(lambda r: s2w(r['signal'], r['confidence']), axis=1)
        daily = SignalConverter._apply_ema_smoothing(daily, symbol, p['ema'], p['decay'], wb, ws)
        daily['weight'] = daily['smoothed_weight']
        aligned = align_to_trading_days(daily, ohlcv.index)
        positions = {r['timestamp']: {r['symbol']: r['weight']} for _, r in aligned.iterrows() if pd.notna(r['weight']) and abs(r['weight']) >= 0.01}
        positions = fill_missing_trading_days(positions, ohlcv.index, symbol=symbol)

        engine = BacktestEngine(initial_cash=1e6, commission=p['commission'])
        engine.add_data(ohlcv, name=symbol)
        engine.add_strategy(DualTrackStrategy, target_positions=positions, printlog=False, allow_short=p['allow_short'])
        result = engine.run()
        model = cf.split('/')[-1].split('_agent.')[0].replace('llm_cache_', '')
        results[model] = {'ret': result.total_return, 'sharpe': result.sharpe_ratio, 'mdd': result.max_drawdown}
    return results

# ===== A股 =====
print("=== A股 CSI300 测试 ===")
csi_best = {'sharpe': -999}
for wb in [0.50, 0.55, 0.60]:
    results = run_one('CSI300', wb, 0.60)
    for m, v in sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True):
        print(f"  wb={wb:.2f}  {m:<28} ret={v['ret']:>7.2%} sharpe={v['sharpe']:>7.3f} mdd={v['mdd']:>7.2%}")
    avg_s = sum(v['sharpe'] for v in results.values()) / len(results)
    avg_r = sum(v['ret'] for v in results.values()) / len(results)
    print(f"  >>> avg: sharpe={avg_s:+.4f} ret={avg_r:+.2%}\n")
    if avg_s > csi_best['sharpe']:
        csi_best = {'sharpe': avg_s, 'ret': avg_r, 'wb': wb}
print(f"*** A股最优: weak_buy={csi_best['wb']:.2f}, avg_sharpe={csi_best['sharpe']:+.4f}, avg_ret={csi_best['ret']:+.2%}\n")

# ===== 美股 =====
print("=== 美股 QQQ 测试 ===")
qqq_best = {'sharpe': -999}
for wb in [0.50, 0.55, 0.60]:
    for ws in [0.50, 0.55, 0.60]:
        results = run_one('QQQ', wb, ws)
        avg_s = sum(v['sharpe'] for v in results.values()) / len(results)
        avg_r = sum(v['ret'] for v in results.values()) / len(results)
        print(f"  wb={wb:.2f} ws={ws:.2f}  avg_sharpe={avg_s:+.4f} avg_ret={avg_r:+.2%}")
        if avg_s > qqq_best['sharpe']:
            qqq_best = {'sharpe': avg_s, 'ret': avg_r, 'wb': wb, 'ws': ws, 'data': results}
print(f"\n*** 美股最优: weak_buy={qqq_best['wb']:.2f}, weak_short={qqq_best['ws']:.2f}, avg_sharpe={qqq_best['sharpe']:+.4f}, avg_ret={qqq_best['ret']:+.2%}")
print(f"\n=== 美股最优详细 ===")
for m, v in sorted(qqq_best['data'].items(), key=lambda x: x[1]['sharpe'], reverse=True):
    print(f"  {m:<28} ret={v['ret']:>7.2%} sharpe={v['sharpe']:>7.3f} mdd={v['mdd']:>7.2%}")
