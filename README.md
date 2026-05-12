[中文](README_zh.md) | [English](README.md) | [Русский](README_ru.md)

---

# DualTrack Quant Research

**Multi-Track Quantitative Trading Comparison Platform** — Strictly compares traditional machine learning (Fitting) and large language models (Semantic Reasoning) in terms of ROI, robustness, and engineering feasibility in quantitative trading, especially during black swan events.

> **Core Positioning**: This project is not a fusion strategy, but an **engineering infrastructure (Testbed)** for comparative experiments, establishing a fair arena where "fitting" and "reasoning" compete under the same market conditions.

---

## Overview

DualTrack is a Python 3.12+ dual-track quantitative backtesting framework that supports:

- **3 Machine Learning Models**: Logistic Regression, LSTM, LightGBM
- **6 Large Language Models**: DeepSeek-V3.2, V4-Flash, R1-8B, Qwen 3.5-397B, GLM-5.1, Gemma-4-31B, with cloud and local deployment
- **SmartPromptAgent**: State-enhanced LLM agent with technical indicators, price history, and closed-loop decision memory
- **2 Markets**: A-Shares (CSI300) and US Stocks (QQQ/NASDAQ-100)
- **Independent Track Backtesting**: Each model generates signals and backtests independently, no signal fusion
- **Multi-Dimensional Evaluation**: Financial metrics (Sharpe, MaxDD, WinRate) + Engineering metrics (Latency, Cost)
- **Complete Master's Thesis**: 4 chapters covering literature review, methodology, empirical results, and conclusions

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    DualTrack Testbed                       │
├────────────────────────────────────────────────────────────┤
│  ML Tracks                             LLM Tracks          │
│  ┌──────┐ ┌────── ┌──────┐    ┌─────────┐ ┌─────────┐     │
│  │ LR   │ │LSTM  │ │ LGB  │    │DeepSeek │ │  Qwen   │     │
│  │(Linear)│(Seq.)│(Ensemble)   │ V3/V4/R1│ │ GLM/Gemma│    │
│  └──┬─── └──┬───┘ └──┬───┘    └────┬────┘ └────┬────┘     │
│     └────────┼────────┘             └─────┬──────┘         │
│              ▼                            ▼                │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │  Signal Converter    │  │  Signal Converter        │    │
│  │  (ML → Target Pos.)  │  │  (LLM → Target Pos.)     │    │
│  └──────────┬───────────┘  └────────────┬─────────────┘    │
│             └─────────────┬─────────────┘                  │
│                           ▼                                │
│              ┌─────────────────────────┐                   │
│              │   Backtrader Engine     │                   │
│              │  Indep. Backtest + P&L  │                   │
│              └─────────────────────────┘                   │
└────────────────────────────────────────────────────────────┘
```

---

## Supported Models

### ML Tracks (Machine Learning)

| Model | Type | Features |
|-------|------|----------|
| `lr` | Logistic Regression | Linear baseline, extremely fast |
| `lstm` | LSTM (PyTorch, MPS-accelerated) | Sequential modeling, Apple Silicon optimized |
| `lgb` | LightGBM | Ensemble learning, feature importance interpretable |

### LLM Tracks (Large Language Models)

| Model | Deployment | Tested Markets | Description |
|-------|-----------|----------------|-------------|
| `deepseek-v3.2` | Cloud (SiliconFlow) | CSI300 + QQQ | DeepSeek V3.2 standard |
| `deepseek-v4-flash` | Cloud (SiliconFlow) | CSI300 + QQQ | DeepSeek V4 Flash lightweight |
| `deepseek-r1-8b` | Local (Ollama) | CSI300 + QQQ | DeepSeek R1 8B distilled reasoning model |
| `qwen3.5-397b` | Cloud (SiliconFlow) | CSI300 + QQQ | Qwen 3.5 397B full version |
| `glm-5.1` | Cloud (SiliconFlow / DashScope) | CSI300 + QQQ | GLM-5.1 |
| `gemma-4-31b` | Cloud (SiliconFlow) | CSI300 + QQQ | Gemma 4 31B open model |

### SmartPromptAgent (State-Enhanced LLM Agent)

The SmartPromptAgent upgrades LLM tracks from "prompt scripting" to a modern agent paradigm while maintaining a single LLM call per trading day for cost efficiency:

- **Technical Indicators**: RSI, MACD, Bollinger Bands, MA alignment, volume-price relationship — pre-computed by Python and injected as natural language summaries
- **Price History**: Recent 5-day price trend table with volume annotations
- **Closed-Loop Memory**: Past decisions with actual market returns auto-filled back, enabling the LLM to see whether its previous calls were right or wrong
- **Zero Look-ahead Bias**: All injected data strictly limited to T-1 and earlier, enforced by `MarketEnricher` with assert guards
- **Cost-Optimized**: ~300 additional tokens per day, still 1 LLM call/trading day (vs. 2-4x for multi-call agent designs)
- **Temperature**: 0.1 (low randomness for reproducible backtests)

```bash
# Run SmartPromptAgent backtest
uv run python main.py run_llm_agent_backtest \
  --symbol CSI300 \
  --start 2020-01-02 \
  --end 2024-12-31 \
  --executor siliconflow \
  --model deepseek-ai/DeepSeek-V3.2
```

> **Local Deployment Device**: MacBook Air M2 (24GB RAM + 512GB SSD)

---

## Thesis

A complete master's thesis has been written based on this project's experiments, organized into 4 chapters:

| Chapter | Content | File |
|---------|---------|------|
| 1 | Literature Review: ML & LLM in quantitative finance | `thesis/sections/1.*` |
| 2 | Methodology: Dual-track architecture design | `thesis/sections/2.*` |
| 3 | Experiments & Results: Cross-market empirical comparison | `thesis/sections/3.*` |
| 4 | Conclusions & Future Work | `thesis/sections/4.*` |

**Architecture Diagrams** (5 figures in `thesis/figures/`):

| Figure | Content |
|--------|---------|
| Fig 1 | Dual-Track System Architecture |
| Fig 2 | Data Processing and Walk-Forward Pipeline |
| Fig 3 | LSTM Cell Gating Mechanism |
| Fig 5 | LLM Prompt Structure and Chain-of-Thought Reasoning |
| Fig 6 | SmartPromptAgent Architecture |

---

## Quick Start

### Installation

**This project uses `uv` as the package manager (pip is not allowed)**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone https://github.com/StarsunCao/DualTrack_Quant_Research.git
cd DualTrack_Quant_Research
uv sync
```

### Running Backtests

```bash
# Single model backtest
uv run python main.py run --track lr --symbol CSI300
uv run python main.py run --track lstm --symbol CSI300
uv run python main.py run --track lgb --symbol QQQ

# LLM model backtest (requires valid cache or API)
uv run python main.py run --track deepseek-v3.2 --symbol CSI300
uv run python main.py run --track glm-5.1 --symbol QQQ

# Full model comparison (requires all LLM caches ready)
uv run python main.py run --track all --compare --symbol CSI300
```

### Building LLM Cache

LLM API calls are expensive; the project supports offline caching with resume capability:

```bash
uv run python main.py cache-build \
  --symbol CSI300 \
  --start 2020-01-02 \
  --end 2024-12-31 \
  --news-file data/raw/csi300_news_combined_2020_2024.csv \
  --executor siliconflow \
  --model deepseek-ai/DeepSeek-V3.2 \
  --temperature 0.1
```

### Training ML Models

```bash
uv run python scripts/train_ml_models.py --symbol CSI300
uv run python scripts/train_ml_models.py --symbol QQQ
uv run python scripts/train_ml_models.py --all  # Train all markets
```

### Evaluation & Visualization

```bash
# Generate evaluation charts from existing backtest results
uv run python main.py evaluate

# Advanced academic evaluation (MAE/MFE, market state segmentation, SHAP attribution)
uv run python main.py evaluate-advanced \
  --output-dir docs/figures \
  --vix-file data/raw/vix_2015_2024.csv
```

### Environment Variables

```bash
# SiliconFlow API (DeepSeek / Qwen / GLM-5 / Gemma cloud)
export SILICONFLOW_API_KEY="your-key-here"

# Alibaba DashScope (GLM-5 alternative endpoint)
export DASHSCOPE_API_KEY="your-key-here"

# Local Ollama service
export OLLAMA_HOST="http://localhost:11434"
```

---

## Project Structure

```
├── main.py                    # CLI entry point
├── pyproject.toml             # Dependency management (uv)
├── thesis/                    # Master's thesis (4 chapters)
│   ├── sections/              # Chapter markdown files
│   ├── figures/               # Architecture diagrams (5 figures)
│   ├── itmo-vkr-template/     # LaTeX template
│   └── thesis_outline.md      # Thesis outline
├── src/
│   ├── data/                  # Data acquisition and alignment
│   │   ├── market_data.py     # OHLCV (akshare, yfinance)
│   │   ├── news_data.py       # News / sentiment data
│   │   └── data_aligner.py    # Time alignment
│   ├── models/
│   │   ├── ml_track/          # ML tracks
│   │   │   ├── features.py    # 55→17 core features (variance + correlation)
│   │   │   └── baselines.py   # LR, LSTM, LightGBM
│   │   ├── llm_track/         # LLM tracks
│   │   │   ├── prompts.py     # A-Shares CoT prompts
│   │   │   ├── us_prompts.py  # US market prompts
│   │   │   ├── agent.py       # Multi-model executors + SmartPromptAgent
│   │   │   ├── memory.py      # Decision memory with closed-loop feedback
│   │   │   └── enricher.py    # Market data enricher (indicators, prices, memory)
│   │   └── model_manager.py
│   ├── orchestrator/          # Signal orchestration
│   │   ├── signal_converter.py
│   │   └── comparator.py
│   ├── execution/             # Backtest execution
│   │   ├── bt_engine.py       # Backtrader wrapper
│   │   ├── base_strategy.py
│   │   ├── a_share_strategy.py
│   │   └── us_market_strategy.py
│   └── evaluation/            # Multi-dimensional evaluation
│       ├── metrics_calculator.py
│       ├── visualizer.py
│       ├── trade_analyzer.py
│       ├── market_state_analyzer.py
│       ├── ml_explainer.py
│       ├── llm_explainer.py
│       └── attribution_comparator.py
├── config/                    # YAML configuration
│   ├── llm_config.yaml
│   ├── ml_config.yaml
│   ├── data_config.yaml
│   └── backtest_config.yaml
├── scripts/                   # Data processing, training, and analysis
│   ├── train_ml_models.py
│   ├── fetch_financial_news.py
│   ├── prepare_us_news.py
│   └── chapter3_analysis.py   # Chapter 3 empirical analysis script
├── models/                    # Pre-trained models (Walk-Forward versions)
├── tests/                     # Unit tests
└── data/                      # Data files (.gitignore)
```

---

## Key Design Principles

### 1. No Look-ahead Bias

Feature engineering uses `shift()` to ensure only historical data is used; signal alignment uses `ffill` forward filling.

### 2. Independent Tracks (No Fusion)

Each model generates signals, converts positions, and backtests independently. No signal fusion ensures fair comparison.

### 3. LLM Offline Caching

LLM API calls are expensive and unstable; the project supports `.jsonl` format offline caching with resume capability.

### 4. Apple Silicon Optimization

PyTorch LSTM models automatically detect and use MPS (Metal Performance Shaders) acceleration.

### 5. Walk-Forward Training

ML models use 912-day training window with 182-day retrain frequency. Feature selection reduces 55 features to 17 core features via variance threshold + correlation dedup.

---

## Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| **Return Capability** | Sharpe Ratio, Sortino Ratio, Calmar Ratio, Total Return |
| **Risk Control** | Maximum Drawdown, drawdown duration, black swan period performance |
| **Trade Quality** | MAE/MFE analysis, entry/hold/exit efficiency |
| **Engineering Metrics** | Inference latency, throughput, cost per alpha signal |
| **Market State** | Performance under Normal / High-Vol / Black-Swan conditions |
| **Interpretability** | ML: SHAP feature attribution; LLM: Reasoning topic analysis |
| **Statistical Robustness** | Jobson-Korkie, Bootstrap, KS tests for performance significance |

---

## Key Empirical Findings

| Finding | Detail |
|---------|--------|
| **ML dominates in A-Shares** | LR: +9.34% vs BuyHold -5.23%, Sharpe 0.172, turnover 0.0019 |
| **LLM competitive in US stocks** | Qwen 3.5-397B: +22.80%, QQQ market with strong Beta Drift (+67.16%) |
| **LLM confidence ≠ accuracy** | Correlation near zero (-0.055 to 0.064) across all 6 models |
| **Conservative bias** | LLMs rarely express high confidence (≥0.80); miss 20.54% of up days (V4-Flash) |
| **Over-interpretation** | All 6 LLMs collectively short on 2020-03-13 (V-shaped rebound day, +8.47%) |
| **No statistical significance** | ML-Ensemble vs LLM-Qwen: Jobson-Korkie p=0.8412, Bootstrap CI [-1.87, 2.06] |

---

## Testing

```bash
uv run pytest tests/ -v
```

Comprehensive test coverage including data acquisition, ML tracks, LLM tracks, signal fusion, backtest engine, and evaluation modules.

---

## Docker Deployment

Docker images are for Linux/CUDA cloud deployment only. Apple Silicon Mac users should use `uv` directly:

```bash
# Apple Silicon Mac (recommended)
uv sync
uv run python main.py run --track lr --symbol CSI300

# Docker (Linux/CUDA)
docker build -t dualtrack-quant .
docker run --rm dualtrack-quant python main.py run --track all --compare --symbol CSI300
```

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Data Layer | ✅ Done | OHLCV acquisition, news data, time alignment |
| Phase 2: ML Tracks | ✅ Done | LR, LSTM, LightGBM + Walk-Forward + feature selection |
| Phase 3: LLM Tracks | ✅ Done | 6 models (DeepSeek V3.2/V4/R1, Qwen, GLM-5.1, Gemma) |
| Phase 4: Orchestrator | ✅ Done | Signal conversion for both tracks |
| Phase 5: Execution Engine | ✅ Done | Backtrader backtest wrapper |
| Phase 6: Evaluation | ✅ Done | Multi-dimensional metrics, visualization |
| Phase 7: CLI & Docker | ✅ Done | Multi-track CLI, containerization |
| Phase 8: Advanced Evaluation | ✅ Done | MAE/MFE, market state segmentation, SHAP attribution |
| Phase 9: Agent Architecture | ✅ Done | SmartPromptAgent: indicators, price history, closed-loop memory |
| Phase 10: Statistical Tests | ✅ Done | Jobson-Korkie, Bootstrap, KS, Chi-square |
| Phase 11: Thesis Writing | ✅ Done | 4 chapters, 5 architecture diagrams, full empirical analysis |

---

## License

MIT License
