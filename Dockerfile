# DualTrack Quant Research - Dockerfile
#
# ⚠️ 重要警告 / IMPORTANT WARNING ⚠️
#
# 由于 Docker for Mac 无法穿透调用 Apple Silicon MPS (Metal Performance Shaders)，
# 对于 M 系列芯片 Mac 用户，建议直接使用本地 uv 虚拟环境运行回测，
# 以确保 LSTM 训练速度和 PyTorch GPU 加速。
#
# Docker 镜像仅用于 Linux/CUDA 云端部署。
#
# For Apple Silicon (M-series) Mac users:
# - MPS acceleration is NOT available inside Docker containers
# - Use local `uv` virtual environment for optimal LSTM training performance
# - This Docker image is intended for Linux/CUDA cloud deployment only

FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（推荐的包管理器）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 复制依赖文件
COPY pyproject.toml uv.lock* ./

# 安装 Python 依赖（CPU 版本 PyTorch，避免 CUDA 依赖）
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu || true
RUN uv pip install --system --no-cache -e .

# 复制项目代码
COPY . .

# 创建必要的目录
RUN mkdir -p docs/figures docs/cache/llm_responses

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 默认命令：显示帮助信息
CMD ["python", "main.py", "--help"]

# ============================================================
# 使用说明 / Usage Instructions
# ============================================================
#
# 构建镜像 / Build Image:
#   docker build -t dualtrack-quant .
#
# 运行回测 / Run Backtest:
#   docker run --rm dualtrack-quant python main.py run --symbol CSI300 --start 2026-01-09 --end 2026-02-28
#
# 生成图表 / Generate Figures:
#   docker run --rm -v $(pwd)/docs/figures:/app/docs/figures dualtrack-quant python main.py evaluate
#
# 注意：对于 Apple Silicon Mac 用户，请直接使用本地环境：
#   uv run python main.py run --symbol CSI300
#
# ============================================================