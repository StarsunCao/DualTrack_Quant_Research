"""
统一模型管理模块。

提供统一的模型保存、加载、版本管理接口，支持 ML 和 LLM 模型。
包含完整的元数据管理和版本追溯功能。
"""

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
from enum import Enum

import joblib
import pandas as pd
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """模型类型枚举。"""
    LOGISTIC_REGRESSION = "logistic_regression"
    LSTM = "lstm"
    LIGHTGBM = "lightgbm"
    LLM_CACHE = "llm_cache"


@dataclass
class ModelMetadata:
    """
    模型元数据类。

    包含模型的完整信息，用于版本追溯和复现。
    """
    model_name: str
    model_type: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"

    # 训练信息
    training_start_time: Optional[str] = None
    training_end_time: Optional[str] = None
    training_samples: int = 0
    feature_count: int = 0

    # 超参数
    hyperparameters: dict = field(default_factory=dict)

    # 评估指标
    metrics: dict = field(default_factory=dict)

    # 环境信息
    git_commit_hash: str = field(default_factory=lambda: ModelManager._get_git_commit())
    python_version: str = field(default_factory=lambda: ModelManager._get_python_version())
    random_seed: Optional[int] = None

    # 特征信息
    feature_names: list[str] = field(default_factory=list)
    target_column: Optional[str] = None

    # 其他
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """从字典创建元数据对象。"""
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class ModelInfo:
    """
    模型信息摘要类。

    用于列出和比较模型。
    """
    name: str
    path: Path
    model_type: str
    created_at: str
    version: str
    metrics_summary: dict = field(default_factory=dict)
    file_size_mb: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "name": self.name,
            "path": str(self.path),
            "model_type": self.model_type,
            "created_at": self.created_at,
            "version": self.version,
            "metrics_summary": self.metrics_summary,
            "file_size_mb": self.file_size_mb,
        }


class ModelManager:
    """
    统一模型管理器。

    提供模型保存、加载、版本管理的统一接口。

    Attributes:
        models_dir: 模型存储根目录
        cache_dir: LLM 缓存目录
    """

    def __init__(self, models_dir: str = "models", cache_dir: str = "docs/cache/llm_responses"):
        """
        初始化模型管理器。

        Args:
            models_dir: 模型存储根目录
            cache_dir: LLM 缓存目录
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"ModelManager 初始化: models_dir={models_dir}, cache_dir={cache_dir}")

    @staticmethod
    def _get_git_commit() -> str:
        """获取当前 Git commit hash。"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _get_python_version() -> str:
        """获取 Python 版本。"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_model_dir(self, symbol: str, model_type: str) -> Path:
        """获取模型目录路径。"""
        model_dir = self.models_dir / f"{symbol.lower()}_{model_type.lower()}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def save_ml_model(
        self,
        model: Any,
        symbol: str,
        model_type: ModelType,
        metadata: Optional[ModelMetadata] = None,
        **kwargs,
    ) -> Path:
        """
        保存 ML 模型。

        Args:
            model: 模型实例
            symbol: 交易标的
            model_type: 模型类型
            metadata: 元数据对象
            **kwargs: 额外的元数据字段

        Returns:
            保存的模型路径
        """
        model_dir = self._get_model_dir(symbol, model_type.value)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型文件
        if model_type == ModelType.LSTM:
            model_path = model_dir / f"model_{timestamp}.pt"
            self._save_lstm_model(model, model_path)
        else:
            model_path = model_dir / f"model_{timestamp}.pkl"
            joblib.dump(model, model_path)

        # 创建或更新元数据
        if metadata is None:
            metadata = ModelMetadata(
                model_name=f"{symbol}_{model_type.value}",
                model_type=model_type.value,
            )

        # 更新元数据
        metadata.updated_at = datetime.now().isoformat()
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        # 保存元数据
        metadata_path = model_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata.to_json())

        # 更新最新版本链接
        latest_model = model_dir / "model_latest.pkl"
        latest_metadata = model_dir / "metadata_latest.json"

        if latest_model.exists():
            latest_model.unlink()
        if latest_metadata.exists():
            latest_metadata.unlink()

        # 创建新版本链接
        latest_model_link = model_dir / "model_latest"
        latest_metadata_link = model_dir / "metadata_latest.json"

        with open(latest_model_link, "w") as f:
            f.write(str(model_path.name))
        with open(latest_metadata_link, "w") as f:
            f.write(metadata.to_json())

        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"模型已保存: {model_path} ({file_size_mb:.2f} MB)")

        return model_path

    def _save_lstm_model(self, model: Any, path: Path) -> None:
        """保存 LSTM 模型。"""
        checkpoint = {
            "model_state_dict": model.model.state_dict(),
            "scaler": model.scaler,
            "config": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
                "sequence_length": model.sequence_length,
                "learning_rate": model.learning_rate,
            },
            "is_fitted": model.is_fitted,
            "train_time": model.train_time,
        }
        torch.save(checkpoint, path)

    def load_ml_model(
        self,
        symbol: str,
        model_type: ModelType,
        version: str = "latest",
    ) -> tuple[Any, ModelMetadata]:
        """
        加载 ML 模型。

        Args:
            symbol: 交易标的
            model_type: 模型类型
            version: 版本号，默认为 "latest"

        Returns:
            (模型实例, 元数据) 元组
        """
        model_dir = self._get_model_dir(symbol, model_type.value)

        if version == "latest":
            # 查找最新版本
            metadata_path = model_dir / "metadata_latest.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"未找到 {symbol} 的 {model_type.value} 模型")

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

            # 读取模型文件名
            model_link = model_dir / "model_latest"
            with open(model_link, "r") as f:
                model_name = f.read().strip()
            model_path = model_dir / model_name
        else:
            model_path = model_dir / f"model_{version}.pkl"
            metadata_path = model_dir / f"metadata_{version}.json"

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        if model_type == ModelType.LSTM:
            from src.models.ml_track.baselines import LSTMModel
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            model = LSTMModel(
                input_dim=checkpoint["config"]["input_dim"],
                hidden_dim=checkpoint["config"]["hidden_dim"],
                num_layers=checkpoint["config"]["num_layers"],
                dropout=checkpoint["config"]["dropout"],
                sequence_length=checkpoint["config"]["sequence_length"],
            )
            model.model = model._build_model()
            model.model.load_state_dict(checkpoint["model_state_dict"])
            model.scaler = checkpoint["scaler"]
            model.is_fitted = checkpoint["is_fitted"]
            model.train_time = checkpoint["train_time"]
        else:
            model = joblib.load(model_path)

        logger.info(f"模型已加载: {model_path} | 版本: {metadata.version} | 创建时间: {metadata.created_at}")

        return model, metadata

    def model_exists(self, symbol: str, model_type: ModelType) -> bool:
        """
        检查模型是否存在。

        Args:
            symbol: 交易标的
            model_type: 模型类型

        Returns:
            是否存在
        """
        model_dir = self._get_model_dir(symbol, model_type.value)
        return (model_dir / "model_latest").exists()

    def list_models(self, symbol: Optional[str] = None) -> list[ModelInfo]:
        """
        列出所有模型。

        Args:
            symbol: 可选，按交易标的过滤

        Returns:
            模型信息列表
        """
        models: list[ModelInfo] = []

        if not self.models_dir.exists():
            return models

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            if symbol and not model_dir.name.startswith(f"{symbol.lower()}_"):
                continue

            metadata_path = model_dir / "metadata_latest.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = ModelMetadata.from_dict(json.load(f))

                # 查找模型文件
                model_link = model_dir / "model_latest"
                if model_link.exists():
                    with open(model_link, "r") as f:
                        model_name = f.read().strip()
                    model_path = model_dir / model_name

                    file_size_mb = 0.0
                    if model_path.exists():
                        file_size_mb = model_path.stat().st_size / (1024 * 1024)

                    models.append(ModelInfo(
                        name=metadata.model_name,
                        path=model_path,
                        model_type=metadata.model_type,
                        created_at=metadata.created_at,
                        version=metadata.version,
                        metrics_summary=metadata.metrics,
                        file_size_mb=file_size_mb,
                    ))

        return sorted(models, key=lambda x: x.created_at, reverse=True)

    def get_model_info(self, symbol: str, model_type: ModelType) -> Optional[ModelInfo]:
        """
        获取模型信息。

        Args:
            symbol: 交易标的
            model_type: 模型类型

        Returns:
            模型信息，如果不存在则返回 None
        """
        model_dir = self._get_model_dir(symbol, model_type.value)
        metadata_path = model_dir / "metadata_latest.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = ModelMetadata.from_dict(json.load(f))

        model_link = model_dir / "model_latest"
        with open(model_link, "r") as f:
            model_name = f.read().strip()
        model_path = model_dir / model_name

        file_size_mb = 0.0
        if model_path.exists():
            file_size_mb = model_path.stat().st_size / (1024 * 1024)

        return ModelInfo(
            name=metadata.model_name,
            path=model_path,
            model_type=metadata.model_type,
            created_at=metadata.created_at,
            version=metadata.version,
            metrics_summary=metadata.metrics,
            file_size_mb=file_size_mb,
        )

    def delete_model(self, symbol: str, model_type: ModelType, version: Optional[str] = None) -> bool:
        """
        删除模型。

        Args:
            symbol: 交易标的
            model_type: 模型类型
            version: 版本号，默认为 None（删除所有版本）

        Returns:
            是否成功删除
        """
        model_dir = self._get_model_dir(symbol, model_type.value)

        if version is None:
            # 删除整个目录
            import shutil
            shutil.rmtree(model_dir)
            logger.info(f"模型已删除: {model_dir}")
            return True
        else:
            # 删除特定版本
            model_path = model_dir / f"model_{version}.pkl"
            metadata_path = model_dir / f"metadata_{version}.json"

            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

            logger.info(f"模型版本 {version} 已删除")
            return True

    def compare_models(self, symbol: str, model_type: ModelType) -> pd.DataFrame:
        """
        比较同一类型的所有模型版本。

        Args:
            symbol: 交易标的
            model_type: 模型类型

        Returns:
            包含比较结果的 DataFrame
        """
        model_dir = self._get_model_dir(symbol, model_type.value)

        versions = []
        for metadata_file in model_dir.glob("metadata_*.json"):
            if metadata_file.name == "metadata_latest.json":
                continue

            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

            versions.append({
                "version": metadata.version,
                "created_at": metadata.created_at,
                "training_samples": metadata.training_samples,
                "metrics": metadata.metrics,
                **metadata.hyperparameters,
            })

        return pd.DataFrame(versions)


# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例。"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


if __name__ == "__main__":
    # 示例用法
    from src.models.ml_track.baselines import LogisticRegressionModel, LSTMModel, LightGBMModel

    print("=" * 60)
    print("ModelManager 示例")
    print("=" * 60)

    manager = ModelManager()

    # 创建示例模型
    print("\n创建示例模型...")
    lr_model = LogisticRegressionModel()

    import numpy as np
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    lr_model.fit(X, y)

    # 保存模型
    print("\n保存模型...")
    metadata = ModelMetadata(
        model_name="CSI300_LR",
        model_type="logistic_regression",
        training_samples=100,
        feature_count=10,
        hyperparameters={"C": 1.0, "max_iter": 1000},
        metrics={"accuracy": 0.85, "f1": 0.82},
        description="沪深300 Logistic Regression 模型",
    )

    model_path = manager.save_ml_model(
        model=lr_model,
        symbol="CSI300",
        model_type=ModelType.LOGISTIC_REGRESSION,
        metadata=metadata,
    )
    print(f"模型保存路径: {model_path}")

    # 列出模型
    print("\n列出所有模型:")
    models = manager.list_models()
    for info in models:
        print(f"  - {info.name} ({info.model_type}) v{info.version}")

    # 加载模型
    print("\n加载模型...")
    loaded_model, loaded_metadata = manager.load_ml_model(
        symbol="CSI300",
        model_type=ModelType.LOGISTIC_REGRESSION,
    )
    print(f"加载的模型类型: {type(loaded_model).__name__}")
    print(f"元数据: {loaded_metadata.to_json()}")

    # 检查模型是否存在
    print("\n检查模型是否存在:")
    print(f"  CSI300 LR: {manager.model_exists('CSI300', ModelType.LOGISTIC_REGRESSION)}")
    print(f"  CSI300 LSTM: {manager.model_exists('CSI300', ModelType.LSTM)}")
