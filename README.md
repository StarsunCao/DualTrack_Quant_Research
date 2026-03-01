# DualTrack Quant Research

双轨制（Dual-Track）量化交易回测框架，核心目标是严格对比传统机器学习（Fitting）与大语言模型智能体（Semantic Reasoning）在量化交易中的 ROI 和鲁棒性，特别是在黑天鹅事件下的表现。

## ⚠️ 重要警告：Apple Silicon 用户必读

**Docker for Mac 无法穿透调用 Apple Silicon MPS (Metal Performance Shaders)。**

对于 M 系列芯片 Mac 用户：
- ❌ **不要**在 Docker 中运行本项目的 LSTM 训练
- ✅ **建议**直接使用本地 `uv` 虚拟环境运行回测
- ✅ Docker 镜像仅用于 Linux/CUDA 云端部署

```bash
# Apple Silicon Mac 推荐使用方式
uv sync
uv run python main.py run --symbol CSI300
```

## Apple Silicon 优化

⚠️ **Important for M-series Mac Users**

Due to Docker for Mac's limitation in accessing Metal Performance Shaders (MPS), **Apple Silicon users should use local `uv` virtual environment instead of Docker** for optimal performance.

### Recommended Setup for M-series Mac

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
uv sync

# 3. Run backtest with MPS acceleration
uv run python main.py run --symbol CSI300 --start 2026-01-09 --end 2026-02-28

# 4. Build LLM cache (optional)
uv run python main.py cache-build --symbol CSI300
```

### Performance Comparison

| Environment | LSTM Training Speed | LLM Inference |
|-------------|---------------------|---------------|
| Local uv + MPS | **~3-5x faster** | Ollama (local) |
| Docker for Mac | CPU only (slow) | Network calls only |

### Docker Usage

Docker is recommended **only for Linux/CUDA environments**:

```bash
# Build image
docker build -t dualtrack-quant .

# Run container
docker run -it --rm dualtrack-quant python main.py --help
```

## 快速开始

### 安装依赖

```bash
# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### CLI 命令

```bash
# 查看帮助
python main.py --help

# 执行完整回测流水线 (Phase 1-6)
python main.py run --symbol CSI300 --start 2026-01-09 --end 2026-02-28

# 重新生成评估图表
python main.py evaluate

# 构建 LLM 离线缓存（支持断点续传）
python main.py cache-build --symbol CSI300 --start 2026-01-09 --end 2026-02-28
```

## 项目结构

```
src/
├── data/              # Phase 1: 数据获取与对齐
├── models/
│   ├── ml_track/      # Phase 2: ML 轨道 (LR, LSTM, LightGBM)
│   └── llm_track/     # Phase 3: LLM 轨道 (Ollama, DeepSeek)
├── orchestrator/      # Phase 4: 双轨信号融合
├── execution/         # Phase 5: Backtrader 回测引擎
└── evaluation/        # Phase 6: 多维度评估与可视化
```

## 双轨融合规则

| 市场状态 | 波动率阈值 | ML 权重 | LLM 权重 | 特殊行为 |
|---------|-----------|--------|---------|---------|
| Normal | < 2% | 70% | 30% | - |
| High Volatility | 2% ~ 5% | 50% | 50% | 增强 LLM 风险提示 |
| Black Swan | > 5% | 0% | 100% | LLM 否决权，强制清仓 |

## 关键指标

### 金融指标
- Sharpe Ratio（夏普比率）
- Maximum Drawdown（最大回撤）
- Win Rate（胜率）
- Sortino Ratio / Calmar Ratio

### 工程指标
- Average Latency（平均推理延迟）
- Throughput（吞吐量）
- Cost-per-Alpha（每个 Alpha 信号成本）

## Docker 部署（仅 Linux/CUDA）

```bash
# 构建镜像
docker build -t dualtrack-quant .

# 运行回测
docker run --rm dualtrack-quant python main.py run --symbol CSI300

# 使用 docker-compose
docker-compose up -d
```

## 环境变量

| 变量名 | 用途 |
|-------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `OLLAMA_HOST` | Ollama 服务地址 |

## License

MIT License