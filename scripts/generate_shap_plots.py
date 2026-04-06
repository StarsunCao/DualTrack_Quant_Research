#!/usr/bin/env python3
"""
SHAP 特征重要性图生成脚本。

生成论文 5.5 节"可解释性归因分析"所需的 SHAP 图表。

用法:
    uv run python scripts/generate_shap_plots.py --symbol CSI300
    uv run python scripts/generate_shap_plots.py --symbol QQQ
    uv run python scripts/generate_shap_plots.py --all
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.utils.logger import get_logger
from src.models.ml_track.features import FeatureEngineer

logger = get_logger(__name__)

# 检查 SHAP 是否可用
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP 未安装。请运行: uv add shap")


def load_model(model_path: str) -> Tuple[Any, dict]:
    """
    加载训练好的模型。

    Args:
        model_path: 模型目录路径。

    Returns:
        (model, metadata) 元组。
    """
    model_dir = Path(model_path)

    # 检查 model_latest 文件（指向实际的模型文件）
    model_latest_path = model_dir / "model_latest"
    if model_latest_path.exists():
        with open(model_latest_path, 'r') as f:
            model_filename = f.read().strip()
        model_file = model_dir / model_filename
    else:
        # 查找最新的模型文件
        pkl_files = list(model_dir.glob("model_*.pkl"))
        pt_files = list(model_dir.glob("model_*.pt"))

        if pkl_files:
            model_file = max(pkl_files, key=lambda x: x.stat().st_mtime)
        elif pt_files:
            model_file = max(pt_files, key=lambda x: x.stat().st_mtime)
        else:
            raise FileNotFoundError(f"No model found in {model_path}")

    # 加载模型
    if model_file.suffix == '.pkl':
        model_wrapper = joblib.load(model_file)
        # 如果是包装类，提取底层模型
        if hasattr(model_wrapper, 'model') and model_wrapper.model is not None:
            model = model_wrapper.model
        else:
            model = model_wrapper
        model_type = "sklearn"
    elif model_file.suffix == '.pt':
        import torch
        model = torch.load(model_file, map_location='cpu')
        model_type = "pytorch"
    else:
        raise ValueError(f"Unknown model format: {model_file.suffix}")

    # 加载元数据
    metadata_latest = model_dir / "metadata_latest.json"
    metadata_files = list(model_dir.glob("metadata_*.json"))

    metadata = {}
    if metadata_latest.exists():
        with open(metadata_latest, 'r') as f:
            metadata = json.load(f)
    elif metadata_files:
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        with open(latest_metadata, 'r') as f:
            metadata = json.load(f)

    logger.info(f"Loaded model from {model_file}, type: {model_type}")
    return model, metadata


def load_market_data(symbol: str) -> pd.DataFrame:
    """
    加载市场数据用于 SHAP 分析。

    Args:
        symbol: 市场代码 ('CSI300' 或 'QQQ')。

    Returns:
        包含 OHLCV 的 DataFrame。
    """
    from src.data.market_data import MarketDataFetcher

    fetcher = MarketDataFetcher()
    if symbol == "CSI300":
        df = fetcher.fetch_csi300(start_date="2020-01-01", end_date="2024-12-31")
    elif symbol == "QQQ":
        df = fetcher.fetch_qqq(start_date="2018-01-01", end_date="2020-07-22")
    else:
        raise ValueError(f"Unknown symbol: {symbol}")

    return df


def prepare_features(df: pd.DataFrame, feature_names: Optional[list] = None) -> Tuple[np.ndarray, list]:
    """
    准备特征矩阵。

    Args:
        df: OHLCV DataFrame。
        feature_names: 指定的特征名称列表（如果提供，只保留这些特征）。

    Returns:
        (X, feature_names) 元组。
    """
    engineer = FeatureEngineer()
    features_df = engineer.compute_all_features(df)

    # 移除 NaN 行
    features_df = features_df.dropna()

    # 如果指定了特征名称，只保留这些特征
    if feature_names is not None:
        # 找出缺失的特征
        missing_features = set(feature_names) - set(features_df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # 用 0 填充缺失的特征
            for feat in missing_features:
                features_df[feat] = 0

        # 只保留指定的特征
        available_features = [f for f in feature_names if f in features_df.columns]
        X = features_df[feature_names].values
        final_feature_names = feature_names
    else:
        # 获取特征名称
        feature_cols = [col for col in features_df.columns if col not in
                       ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date']]
        X = features_df[feature_cols].values
        final_feature_names = feature_cols

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(final_feature_names)}")

    return X, final_feature_names


def generate_shap_summary_plot(
    model: Any,
    X: np.ndarray,
    feature_names: list,
    output_path: str,
    max_display: int = 20,
    figsize: Tuple[int, int] = (12, 10)
) -> bool:
    """
    生成 SHAP Summary Plot。

    Args:
        model: 训练好的模型。
        X: 特征矩阵。
        feature_names: 特征名称列表。
        output_path: 输出文件路径。
        max_display: 最多显示的特征数量。
        figsize: 图表尺寸。

    Returns:
        是否成功生成图表。
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP 未安装，无法生成图表")
        return False

    import shap

    logger.info("Creating SHAP explainer...")

    # 根据模型类型选择 explainer
    model_type = type(model).__name__.lower()

    try:
        if "lightgbm" in model_type or "lgbm" in model_type or "booster" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # 二分类取正类
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif "logistic" in model_type:
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        elif "lgbm" in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # 尝试 TreeExplainer
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            except Exception:
                # 使用 KernelExplainer（较慢）
                logger.warning("Using KernelExplainer (slower)...")
                explainer = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    shap.kmeans(X, 50)  # 使用 K-means 采样加速
                )
                shap_values = explainer.shap_values(X[:500])  # 限制样本数
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

        logger.info(f"SHAP values shape: {shap_values.shape}")

        # 创建 DataFrame 以获得更好的可视化
        X_df = pd.DataFrame(X[:len(shap_values)], columns=feature_names)

        # 绘制 SHAP Summary Plot
        plt.figure(figsize=figsize)
        shap.summary_plot(
            shap_values,
            X_df,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_size=figsize
        )

        plt.title("SHAP Feature Importance Analysis", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
        plt.tight_layout()

        # 保存图表
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"SHAP Summary Plot saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating SHAP plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_shap_bar_plot(
    model: Any,
    X: np.ndarray,
    feature_names: list,
    output_path: str,
    max_display: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> bool:
    """
    生成 SHAP 条形图（平均绝对贡献）。

    Args:
        model: 训练好的模型。
        X: 特征矩阵。
        feature_names: 特征名称列表。
        output_path: 输出文件路径。
        max_display: 最多显示的特征数量。
        figsize: 图表尺寸。

    Returns:
        是否成功生成图表。
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP 未安装，无法生成图表")
        return False

    import shap

    try:
        # 根据模型类型选择 explainer
        model_type = type(model).__name__.lower()

        if "lightgbm" in model_type or "lgbm" in model_type or "booster" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif "logistic" in model_type:
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        # 绘制条形图
        plt.figure(figsize=figsize)
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False
        )

        plt.title("Mean Absolute SHAP Values", fontsize=14, fontweight='bold')
        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"SHAP Bar Plot saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating SHAP bar plot: {e}")
        return False


def generate_feature_importance_comparison(
    models_data: dict,
    output_path: str,
    top_n: int = 15,
    figsize: Tuple[int, int] = (14, 10)
) -> bool:
    """
    生成多模型特征重要性对比图。

    Args:
        models_data: {model_name: (importance_dict)} 字典。
        output_path: 输出文件路径。
        top_n: 显示 top N 特征。
        figsize: 图表尺寸。

    Returns:
        是否成功生成图表。
    """
    if not models_data:
        logger.error("No model data provided")
        return False

    # 创建对比 DataFrame
    all_features = set()
    for importance_dict in models_data.values():
        all_features.update(importance_dict.keys())

    # 取所有模型的 top 特征
    top_features = set()
    for importance_dict in models_data.values():
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features.update([f[0] for f in sorted_features[:top_n]])

    # 创建对比图
    fig, axes = plt.subplots(1, len(models_data), figsize=figsize)
    if len(models_data) == 1:
        axes = [axes]

    for idx, (model_name, importance_dict) in enumerate(models_data.items()):
        # 取 top N 特征
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in sorted_features][::-1]  # 反转以从下往上绘制
        values = [f[1] for f in sorted_features][::-1]

        ax = axes[idx]
        bars = ax.barh(features, values, color='steelblue', edgecolor='navy')

        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Feature importance comparison saved to {output_path}")
    return True


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance plots")
    parser.add_argument("--symbol", type=str, default="CSI300",
                       choices=["CSI300", "QQQ"],
                       help="Market symbol")
    parser.add_argument("--all", action="store_true",
                       help="Generate plots for all markets")
    parser.add_argument("--output-dir", type=str, default="docs/figures",
                       help="Output directory for plots")

    args = parser.parse_args()

    if args.all:
        symbols = ["CSI300", "QQQ"]
    else:
        symbols = [args.symbol]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*60}")

        # 加载市场数据
        logger.info("Loading market data...")
        df = load_market_data(symbol)

        # 为每个模型生成 SHAP 图
        model_types = ["lightgbm", "logistic_regression"]
        models_data = {}

        for model_type in model_types:
            model_path = f"models/{symbol.lower()}_{model_type}"

            if not Path(model_path).exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            try:
                # 加载模型
                model, metadata = load_model(model_path)

                # 从元数据获取特征名称
                model_feature_names = metadata.get('feature_names', None)

                # 准备特征（使用模型训练时的特征）
                logger.info("Preparing features...")
                X, feature_names = prepare_features(df, model_feature_names)

                # 生成 SHAP Summary Plot
                output_name = f"shap_summary_{symbol.lower()}_{model_type}.png"
                output_path = output_dir / output_name

                success = generate_shap_summary_plot(
                    model=model,
                    X=X,
                    feature_names=feature_names,
                    output_path=str(output_path),
                    max_display=20
                )

                if success:
                    logger.info(f"Generated: {output_path}")

                # 生成条形图
                output_name = f"shap_bar_{symbol.lower()}_{model_type}.png"
                output_path = output_dir / output_name

                success = generate_shap_bar_plot(
                    model=model,
                    X=X,
                    feature_names=feature_names,
                    output_path=str(output_path),
                    max_display=15
                )

                # 获取特征重要性用于对比
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    importance_dict = dict(zip(feature_names, np.abs(model.coef_[0])))
                else:
                    importance_dict = {}

                if importance_dict:
                    models_data[model_type] = importance_dict

            except Exception as e:
                logger.error(f"Error processing {model_type}: {e}")
                continue

        # 生成多模型对比图
        if len(models_data) > 1:
            output_path = output_dir / f"feature_importance_comparison_{symbol.lower()}.png"
            generate_feature_importance_comparison(
                models_data=models_data,
                output_path=str(output_path),
                top_n=15
            )

    # 生成最终用于论文的主图（CSI300 LightGBM）
    logger.info("\nGenerating main SHAP plot for thesis...")
    main_output = output_dir / "ml_shap_importance.png"

    # 如果 CSI300 LightGBM 的图已生成，复制到主位置
    csi300_lgb_path = output_dir / f"shap_summary_csi300_lightgbm.png"
    if csi300_lgb_path.exists():
        import shutil
        shutil.copy(csi300_lgb_path, main_output)
        logger.info(f"Main SHAP plot saved to: {main_output}")
    else:
        logger.warning("CSI300 LightGBM SHAP plot not found")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()