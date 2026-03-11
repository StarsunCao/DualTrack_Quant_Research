#!/usr/bin/env python
"""
ML 模型训练脚本。

训练并持久化 ML 模型（LR、LightGBM、LSTM）用于高级评估。

使用方法:
    # 训练 A 股模型
    uv run python scripts/train_ml_models.py --symbol CSI300

    # 训练美股模型
    uv run python scripts/train_ml_models.py --symbol QQQ

    # 训练所有模型
    uv run python scripts/train_ml_models.py --all
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import click
import numpy as np
import pandas as pd

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.models.ml_track.features import FeatureEngineer
from src.models.ml_track.baselines import (
    LogisticRegressionModel,
    LightGBMModel,
    LSTMModel,
    MLStrategyPortfolio,
)
from src.models.model_manager import ModelManager, ModelMetadata, ModelType
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 训练配置
# ============================================================================
TRAINING_CONFIG = {
    "CSI300": {
        "train_data": "data/raw/csi300_train_2015_2019.csv",
        "test_data": "data/raw/real_csi300_5y.csv",
        "train_start": "2015-01-05",
        "train_end": "2018-12-31",
        "val_start": "2019-01-01",
        "val_end": "2019-12-31",
        "test_start": "2020-01-02",
        "target_sharpe": 1.0,
        "target_max_dd": 0.20,
    },
    "QQQ": {
        "train_data": "data/raw/qqq_train_2015_2017.csv",
        "test_data": "data/raw/real_qqq_5y.csv",
        "train_start": "2015-01-01",
        "train_end": "2016-12-31",
        "val_start": "2017-01-01",
        "val_end": "2017-12-31",
        "test_start": "2018-01-02",
        "target_sharpe": 1.0,
        "target_max_dd": 0.25,
    },
}

# 模型超参数
MODEL_PARAMS = {
    "LogisticRegression": {
        "C": 1.0,
        "max_iter": 1000,
    },
    "LightGBM": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
    },
    "LSTM": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 50,
        "sequence_length": 20,
        "learning_rate": 0.001,
    },
}


# ============================================================================
# 训练函数
# ============================================================================
def load_and_prepare_data(
    symbol: str,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载并准备训练数据。

    Returns:
        (train_df, val_df, test_df) 元组
    """
    click.echo(f"\n[1/4] 加载 {symbol} 数据...")

    # 加载训练数据
    train_path = Path(config["train_data"])
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_path}")

    train_ohlcv = pd.read_csv(train_path, parse_dates=["date"])
    train_ohlcv.set_index("date", inplace=True)

    # 加载测试数据（包含验证期）
    test_path = Path(config["test_data"])
    if test_path.exists():
        test_ohlcv = pd.read_csv(test_path, parse_dates=["date"])
        test_ohlcv.set_index("date", inplace=True)
    else:
        click.echo(f"  警告: 测试数据文件不存在 {test_path}，仅使用训练数据")
        test_ohlcv = pd.DataFrame()

    # 切分验证集
    val_start = pd.to_datetime(config["val_start"])
    val_end = pd.to_datetime(config["val_end"])

    if not test_ohlcv.empty:
        val_mask = (test_ohlcv.index >= val_start) & (test_ohlcv.index <= val_end)
        val_ohlcv = test_ohlcv[val_mask].copy()
    else:
        val_ohlcv = pd.DataFrame()

    click.echo(f"  训练数据: {train_ohlcv.index.min().date()} ~ {train_ohlcv.index.max().date()} ({len(train_ohlcv)} 天)")
    if not val_ohlcv.empty:
        click.echo(f"  验证数据: {val_ohlcv.index.min().date()} ~ {val_ohlcv.index.max().date()} ({len(val_ohlcv)} 天)")

    return train_ohlcv, val_ohlcv, test_ohlcv


def compute_features(
    train_ohlcv: pd.DataFrame,
    val_ohlcv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    计算特征。

    Returns:
        (train_features, val_features, feature_names) 元组
    """
    click.echo("\n[2/4] 计算技术因子...")

    engineer = FeatureEngineer()

    # 计算训练特征
    train_features = engineer.compute_all_features(train_ohlcv, drop_na=True)
    train_features = engineer.create_target(train_features.copy(), forward_period=1)
    train_features = train_features.dropna()

    # 计算验证特征
    val_features = pd.DataFrame()
    if not val_ohlcv.empty:
        val_features = engineer.compute_all_features(val_ohlcv, drop_na=True)
        val_features = engineer.create_target(val_features.copy(), forward_period=1)
        val_features = val_features.dropna()

    # 获取特征列
    exclude_cols = {"target_label", "target_return", "symbol"}
    feature_names = [c for c in train_features.columns if c not in exclude_cols]

    click.echo(f"  特征数量: {len(feature_names)}")
    click.echo(f"  训练样本: {len(train_features)}")
    if not val_features.empty:
        click.echo(f"  验证样本: {len(val_features)}")

    return train_features, val_features, feature_names


def train_models(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    feature_names: list[str],
    symbol: str,
    save_models: bool = True,
) -> Dict[str, Any]:
    """
    训练所有模型。

    Returns:
        包含模型和指标的字典
    """
    click.echo("\n[3/4] 训练模型...")

    # 准备数据
    X_train = train_features[feature_names].values
    y_train = train_features["target_label"].values

    X_val = None
    y_val = None
    if not val_features.empty:
        X_val = val_features[feature_names].values
        y_val = val_features["target_label"].values

    results = {}
    manager = ModelManager() if save_models else None

    # 1. Logistic Regression
    click.echo("\n  [1/3] Logistic Regression...")
    lr_params = MODEL_PARAMS["LogisticRegression"]
    lr_model = LogisticRegressionModel(
        max_iter=lr_params["max_iter"],
        C=lr_params["C"],
    )
    lr_model.fit(X_train, y_train)

    lr_metrics = lr_model.evaluate(X_val, y_val) if X_val is not None else None
    results["LogisticRegression"] = {
        "model": lr_model,
        "metrics": lr_metrics,
    }

    if save_models and manager is not None:
        metadata = ModelMetadata(
            model_name=f"{symbol}_LogisticRegression",
            model_type="logistic_regression",
            training_samples=len(X_train),
            feature_count=len(feature_names),
            hyperparameters=lr_params,
            metrics=lr_metrics.to_dict() if lr_metrics else {},
            feature_names=feature_names,
        )
        manager.save_ml_model(lr_model, symbol, ModelType.LOGISTIC_REGRESSION, metadata)

    # 2. LightGBM
    click.echo("\n  [2/3] LightGBM...")
    lgb_params = MODEL_PARAMS["LightGBM"]
    lgb_model = LightGBMModel(
        n_estimators=lgb_params["n_estimators"],
        max_depth=lgb_params["max_depth"],
        learning_rate=lgb_params["learning_rate"],
    )
    lgb_model.fit(X_train, y_train)

    lgb_metrics = lgb_model.evaluate(X_val, y_val) if X_val is not None else None
    results["LightGBM"] = {
        "model": lgb_model,
        "metrics": lgb_metrics,
    }

    if save_models and manager is not None:
        metadata = ModelMetadata(
            model_name=f"{symbol}_LightGBM",
            model_type="lightgbm",
            training_samples=len(X_train),
            feature_count=len(feature_names),
            hyperparameters=lgb_params,
            metrics=lgb_metrics.to_dict() if lgb_metrics else {},
            feature_names=feature_names,
        )
        manager.save_ml_model(lgb_model, symbol, ModelType.LIGHTGBM, metadata)

    # 3. LSTM
    click.echo("\n  [3/3] LSTM...")
    lstm_params = MODEL_PARAMS["LSTM"]
    lstm_model = LSTMModel(
        input_dim=len(feature_names),
        hidden_dim=lstm_params["hidden_dim"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"],
        epochs=lstm_params["epochs"],
        sequence_length=lstm_params["sequence_length"],
        learning_rate=lstm_params["learning_rate"],
    )
    lstm_model.fit(X_train, y_train)

    lstm_metrics = lstm_model.evaluate(X_val, y_val) if X_val is not None else None
    results["LSTM"] = {
        "model": lstm_model,
        "metrics": lstm_metrics,
    }

    if save_models and manager is not None:
        metadata = ModelMetadata(
            model_name=f"{symbol}_LSTM",
            model_type="lstm",
            training_samples=len(X_train),
            feature_count=len(feature_names),
            hyperparameters=lstm_params,
            metrics=lstm_metrics.to_dict() if lstm_metrics else {},
            feature_names=feature_names,
        )
        manager.save_ml_model(lstm_model, symbol, ModelType.LSTM, metadata)

    return results


def print_summary(results: Dict[str, Any], symbol: str) -> None:
    """打印训练结果摘要。"""
    click.echo("\n[4/4] 训练结果摘要")
    click.echo("=" * 70)
    click.echo(f"  标的: {symbol}")
    click.echo("=" * 70)

    click.echo(f"\n  {'模型':<20} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1':>10} {'训练时间':>12}")
    click.echo("  " + "-" * 72)

    for model_name, result in results.items():
        metrics = result["metrics"]
        if metrics:
            click.echo(
                f"  {model_name:<20} "
                f"{metrics.accuracy:>10.4f} "
                f"{metrics.precision:>10.4f} "
                f"{metrics.recall:>10.4f} "
                f"{metrics.f1:>10.4f} "
                f"{metrics.train_time_sec:>10.2f}s"
            )
        else:
            click.echo(f"  {model_name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    click.echo("=" * 70)
    click.echo("\n  模型已保存至: models/{symbol.lower()}_<model_type>/")
    click.echo("  可使用 ModelManager.load_ml_model() 加载模型")


# ============================================================================
# CLI 入口
# ============================================================================
@click.command()
@click.option("--symbol", "-s", type=click.Choice(["CSI300", "QQQ"]), default="CSI300",
              help="训练标的")
@click.option("--all", "-a", "train_all", is_flag=True, help="训练所有标的")
@click.option("--no-save", is_flag=True, help="不保存模型（仅测试）")
def main(symbol: str, train_all: bool, no_save: bool) -> None:
    """
    训练 ML 模型并持久化。

    示例:
        uv run python scripts/train_ml_models.py --symbol CSI300
        uv run python scripts/train_ml_models.py --all
    """
    click.echo("=" * 70)
    click.echo("  ML 模型训练脚本")
    click.echo("=" * 70)

    symbols = ["CSI300", "QQQ"] if train_all else [symbol]

    for sym in symbols:
        if sym not in TRAINING_CONFIG:
            click.echo(f"  警告: 未知的标的 {sym}，跳过")
            continue

        config = TRAINING_CONFIG[sym]

        try:
            # 1. 加载数据
            train_ohlcv, val_ohlcv, test_ohlcv = load_and_prepare_data(sym, config)

            # 2. 计算特征
            train_features, val_features, feature_names = compute_features(
                train_ohlcv, val_ohlcv
            )

            # 3. 训练模型
            results = train_models(
                train_features, val_features, feature_names,
                symbol=sym,
                save_models=not no_save,
            )

            # 4. 打印摘要
            print_summary(results, sym)

        except FileNotFoundError as e:
            click.echo(f"  错误: {e}")
            continue
        except Exception as e:
            click.echo(f"  训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    click.echo("\n训练完成!")


if __name__ == "__main__":
    main()