"""
机器学习基线模型模块。

实现多个基线模型进行涨跌分类预测，包括：
- Logistic Regression (Baseline)
- LSTM (Sequence Rival) - Apple Silicon MPS 优化
- LightGBM (SOTA Rival)
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 检测设备并设置优先级：MPS (Apple Silicon) > CUDA > CPU
def get_best_device() -> torch.device:
    """
    获取最佳可用设备，优先使用 MPS (Apple Silicon) 或 CUDA。

    Returns:
        torch.device: 最佳可用设备。
    """
    if torch.backends.mps.is_available():
        print("✅ 使用 MPS (Metal Performance Shaders) 设备 - Apple Silicon 优化")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ 使用 CUDA 设备")
        return torch.device("cuda")
    else:
        print("⚠️ 使用 CPU 设备")
        return torch.device("cpu")


# 全局设备变量
DEVICE = get_best_device()


@dataclass
class ModelMetrics:
    """模型评估指标数据类。"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    inference_time_ms: float
    train_time_sec: float

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "inference_time_ms": self.inference_time_ms,
            "train_time_sec": self.train_time_sec,
        }


class BaseModel(ABC):
    """基线模型抽象基类。"""

    def __init__(self, model_name: str) -> None:
        """
        初始化模型。

        Args:
            model_name: 模型名称。
        """
        self.model_name = model_name
        self.is_fitted = False
        self.train_time = 0.0

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "BaseModel":
        """
        训练模型。

        Args:
            X: 特征矩阵。
            y: 目标标签。

        Returns:
            训练后的模型实例。
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签。

        Args:
            X: 特征矩阵。

        Returns:
            预测标签数组。
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率。

        Args:
            X: 特征矩阵。

        Returns:
            预测概率数组。
        """
        pass

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ModelMetrics:
        """
        评估模型性能。

        Args:
            X: 特征矩阵。
            y: 真实标签。

        Returns:
            模型评估指标。
        """
        if not self.is_fitted:
            raise ValueError(f"模型 {self.model_name} 尚未训练")

        # 测量推理时间
        start_time = time.time()
        y_pred = self.predict(X)
        inference_time = (time.time() - start_time) * 1000  # 毫秒

        # 计算评估指标
        metrics = ModelMetrics(
            model_name=self.model_name,
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1=f1_score(y, y_pred, zero_division=0),
            inference_time_ms=inference_time,
            train_time_sec=self.train_time,
        )

        return metrics


class LogisticRegressionModel(BaseModel):
    """Logistic Regression 基线模型。"""

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """
        初始化 Logistic Regression 模型。

        Args:
            max_iter: 最大迭代次数。
            C: 正则化强度的倒数。
            random_state: 随机种子。
        """
        super().__init__("LogisticRegression")
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "LogisticRegressionModel":
        """
        训练 Logistic Regression 模型。

        Args:
            X: 特征矩阵。
            y: 目标标签。

        Returns:
            训练后的模型实例。
        """
        start_time = time.time()

        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 训练模型
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

        self.train_time = time.time() - start_time
        self.is_fitted = True
        print(f"  LogisticRegression 训练完成，耗时: {self.train_time:.2f}秒")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签。"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """
        获取特征重要性。

        Args:
            feature_names: 特征名称列表。

        Returns:
            包含特征重要性的 DataFrame。
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未训练")

        importance = pd.DataFrame({
            "feature": feature_names,
            "coefficient": self.model.coef_[0],
        })
        importance["abs_coefficient"] = importance["coefficient"].abs()
        importance = importance.sort_values("abs_coefficient", ascending=False)

        return importance


class LSTMModel(BaseModel):
    """
    LSTM 序列模型 - Apple Silicon MPS 优化。

    使用长短期记忆网络处理时间序列数据进行涨跌预测。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        sequence_length: int = 20,
    ) -> None:
        """
        初始化 LSTM 模型。

        Args:
            input_dim: 输入特征维度。
            hidden_dim: 隐藏层维度。
            num_layers: LSTM 层数。
            dropout: Dropout 比率。
            learning_rate: 学习率。
            batch_size: 批次大小。
            epochs: 训练轮数。
            sequence_length: 序列长度。
        """
        super().__init__("LSTM")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[nn.Module] = None

    def _build_model(self) -> nn.Module:
        """构建 LSTM 模型。"""
        class LSTMNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                lstm_out, _ = self.lstm(x)
                out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
                return self.fc(out)

        return LSTMNet(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)

    @staticmethod
    def create_sequences(
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本。

        Args:
            X: 特征矩阵。
            y: 目标标签。
            sequence_length: 序列长度。

        Returns:
            序列特征和标签元组。
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "LSTMModel":
        """
        训练 LSTM 模型。

        Args:
            X: 特征矩阵。
            y: 目标标签。

        Returns:
            训练后的模型实例。
        """
        start_time = time.time()

        # 设置随机种子确保可复现性
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(42)
        elif torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 创建序列
        X_seq, y_seq = self.create_sequences(X_scaled, y, self.sequence_length)

        if len(X_seq) == 0:
            raise ValueError(f"数据量不足，无法创建长度为 {self.sequence_length} 的序列")

        # 转换为 PyTorch 张量
        X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(DEVICE)

        # 构建模型
        self.model = self._build_model().to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # 训练
        self.model.train()
        progress_bar = tqdm(range(self.epochs), desc="LSTM Training", unit="epoch")
        for epoch in progress_bar:
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.debug(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

        self.train_time = time.time() - start_time
        self.is_fitted = True
        progress_bar.close()
        logger.info(f"LSTM 训练完成，耗时: {self.train_time:.2f}秒，设备: {DEVICE}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签。"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未训练")

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 如果数据不足以创建序列，返回默认值
        if len(X_scaled) <= self.sequence_length:
            return np.full(len(X), 0.5)

        # 创建序列
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)), self.sequence_length)

        # 检查是否有有效序列
        if len(X_seq) == 0:
            return np.full(len(X), 0.5)

        X_tensor = torch.FloatTensor(X_seq)
        if hasattr(self, 'model') and self.model is not None:
            X_tensor = X_tensor.to(next(self.model.parameters()).device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            # 防止 squeeze 过度降维
            if outputs.dim() == 2:
                outputs = outputs.squeeze(-1)
            elif outputs.dim() == 3:
                outputs = outputs.squeeze(-1).squeeze(-1)

            outputs = outputs.cpu().numpy().flatten()

        # 填充前面的缺失值（无法形成序列的数据点）
        full_proba = np.full(len(X), 0.5)
        full_proba[self.sequence_length:] = outputs[:len(full_proba) - self.sequence_length]

        return full_proba


class LightGBMModel(BaseModel):
    """LightGBM 梯度提升模型。"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        verbose: int = -1,
    ) -> None:
        """
        初始化 LightGBM 模型。

        Args:
            n_estimators: 树的数量。
            max_depth: 最大深度。
            learning_rate: 学习率。
            random_state: 随机种子。
            verbose: 日志详细程度。
        """
        super().__init__("LightGBM")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "LightGBMModel":
        """
        训练 LightGBM 模型。

        Args:
            X: 特征矩阵。
            y: 目标标签。

        Returns:
            训练后的模型实例。
        """
        import lightgbm as lgb

        start_time = time.time()

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        self.train_time = time.time() - start_time
        self.is_fitted = True
        print(f"  LightGBM 训练完成，耗时: {self.train_time:.2f}秒")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签。"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """
        获取特征重要性。

        Args:
            feature_names: 特征名称列表。

        Returns:
            包含特征重要性的 DataFrame。
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未训练")

        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_,
        })
        importance = importance.sort_values("importance", ascending=False)
        importance = importance.reset_index(drop=True)

        return importance


class MLStrategyPortfolio:
    """
    机器学习策略组合类。

    集成多个基线模型进行涨跌预测，输出标准化的信号 DataFrame。

    Attributes:
        models: 模型字典。
        feature_names: 特征名称列表。
        metrics: 模型评估指标字典。
    """

    def __init__(
        self,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        lstm_epochs: int = 30,
        lstm_sequence_length: int = 20,
        lgb_n_estimators: int = 100,
        lgb_max_depth: int = 6,
    ) -> None:
        """
        初始化策略组合。

        Args:
            lstm_hidden_dim: LSTM 隐藏层维度。
            lstm_num_layers: LSTM 层数。
            lstm_epochs: LSTM 训练轮数。
            lstm_sequence_length: LSTM 序列长度。
            lgb_n_estimators: LightGBM 树的数量。
            lgb_max_depth: LightGBM 最大深度。
        """
        self.models: dict[str, BaseModel] = {}
        self.feature_names: list[str] = []
        self.metrics: dict[str, ModelMetrics] = {}
        self._lstm_config = {
            "hidden_dim": lstm_hidden_dim,
            "num_layers": lstm_num_layers,
            "epochs": lstm_epochs,
            "sequence_length": lstm_sequence_length,
        }
        self._lgb_config = {
            "n_estimators": lgb_n_estimators,
            "max_depth": lgb_max_depth,
        }

    def fit(
        self,
        features_df: pd.DataFrame,
        target_col: str = "target_label",
        feature_cols: Optional[list[str]] = None,
        test_size: float = 0.2,
    ) -> "MLStrategyPortfolio":
        """
        训练所有模型。

        Args:
            features_df: 包含特征和目标的 DataFrame。
            target_col: 目标列名。
            feature_cols: 特征列名列表，如果为 None 则自动检测。
            test_size: 测试集比例。

        Returns:
            训练后的策略组合实例。
        """
        print("=" * 60)
        print("  开始训练 ML 策略组合")
        print("=" * 60)

        # 准备数据
        if target_col not in features_df.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在")

        # 确定特征列
        exclude_cols = {target_col, "target_return", "symbol"}
        if feature_cols is None:
            feature_cols = [
                col for col in features_df.columns
                if col not in exclude_cols and not col.startswith("target")
            ]
        self.feature_names = feature_cols

        # 提取特征和目标
        df_clean = features_df.dropna(subset=feature_cols + [target_col])
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        # 时间序列分割（保持时间顺序）
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\n数据集信息:")
        print(f"  - 总样本数: {len(X)}")
        print(f"  - 训练集: {len(X_train)}")
        print(f"  - 测试集: {len(X_test)}")
        print(f"  - 特征数: {len(feature_cols)}")
        print(f"  - 正样本比例: {y.mean():.2%}")

        # 1. 训练 Logistic Regression
        print("\n[1/3] 训练 Logistic Regression...")
        lr_model = LogisticRegressionModel()
        lr_model.fit(X_train, y_train)
        self.models["LogisticRegression"] = lr_model
        self.metrics["LogisticRegression"] = lr_model.evaluate(X_test, y_test)

        # 2. 训练 LSTM
        print("\n[2/3] 训练 LSTM...")
        lstm_model = LSTMModel(
            input_dim=len(feature_cols),
            **self._lstm_config,
        )
        lstm_model.fit(X_train, y_train)
        self.models["LSTM"] = lstm_model
        self.metrics["LSTM"] = lstm_model.evaluate(X_test, y_test)

        # 3. 训练 LightGBM
        print("\n[3/3] 训练 LightGBM...")
        lgb_model = LightGBMModel(**self._lgb_config)
        lgb_model.fit(X_train, y_train)
        self.models["LightGBM"] = lgb_model
        self.metrics["LightGBM"] = lgb_model.evaluate(X_test, y_test)

        # 打印评估结果
        self._print_metrics_summary()

        return self

    def _print_metrics_summary(self) -> None:
        """打印模型评估指标汇总。"""
        print("\n" + "=" * 60)
        print("  模型评估指标汇总")
        print("=" * 60)
        print(f"{'模型':<20} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1':>10} {'推理(ms)':>10}")
        print("-" * 60)
        for name, metrics in self.metrics.items():
            print(
                f"{name:<20} "
                f"{metrics.accuracy:>10.4f} "
                f"{metrics.precision:>10.4f} "
                f"{metrics.recall:>10.4f} "
                f"{metrics.f1:>10.4f} "
                f"{metrics.inference_time_ms:>10.2f}"
            )
        print("=" * 60)

    def predict(
        self,
        features_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        symbol: str = "UNKNOWN",
    ) -> pd.DataFrame:
        """
        使用所有模型生成预测信号。

        Args:
            features_df: 包含特征的 DataFrame。
            feature_cols: 特征列名列表，如果为 None 则使用训练时的特征。
            symbol: 股票/资产代码。

        Returns:
            标准化的信号 DataFrame，包含列：
            - timestamp: 时间戳
            - symbol: 股票代码
            - model_name: 模型名称
            - signal_strength_0_to_1: 信号强度 (0-1)
        """
        if not self.models:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        # 确定特征列
        if feature_cols is None:
            feature_cols = self.feature_names

        # 提取特征
        df_clean = features_df.dropna(subset=feature_cols)
        X = df_clean[feature_cols].values

        # 获取时间戳
        if isinstance(df_clean.index, pd.DatetimeIndex):
            timestamps = df_clean.index.to_list()
        else:
            timestamps = df_clean.index.tolist()

        # 生成预测
        signals: list[dict] = []
        for model_name, model in self.models.items():
            proba = model.predict_proba(X)

            for i, (ts, prob) in enumerate(zip(timestamps, proba)):
                signals.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "model_name": model_name,
                    "signal_strength_0_to_1": float(prob),
                })

        # 创建 DataFrame
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values(["timestamp", "model_name"]).reset_index(drop=True)

        return signals_df

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取模型特征重要性（如果可用）。

        Returns:
            包含特征重要性的 DataFrame。
        """
        importance_list: list[pd.DataFrame] = []

        for model_name in ["LogisticRegression", "LightGBM"]:
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, "get_feature_importance"):
                    df = model.get_feature_importance(self.feature_names)
                    df["model"] = model_name
                    importance_list.append(df)

        if importance_list:
            return pd.concat(importance_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_metrics(self) -> pd.DataFrame:
        """
        获取所有模型的评估指标。

        Returns:
            包含评估指标的 DataFrame。
        """
        metrics_list = [m.to_dict() for m in self.metrics.values()]
        return pd.DataFrame(metrics_list)

    def save_models(self, path: str) -> None:
        """
        保存所有模型。

        Args:
            path: 保存路径前缀。
        """
        import joblib

        for model_name, model in self.models.items():
            model_path = f"{path}_{model_name.lower()}.pkl"
            if isinstance(model, LSTMModel):
                # LSTM 需要特殊处理
                import torch
                torch.save({
                    "model_state_dict": model.model.state_dict(),
                    "scaler": model.scaler,
                    "config": {
                        "input_dim": model.input_dim,
                        "hidden_dim": model.hidden_dim,
                        "num_layers": model.num_layers,
                        "dropout": model.dropout,
                        "sequence_length": model.sequence_length,
                    },
                }, model_path)
            else:
                joblib.dump(model, model_path)
        print(f"模型已保存至: {path}_*.pkl")

    def load_models(self, path: str, feature_dim: int) -> None:
        """
        加载所有模型。

        Args:
            path: 保存路径前缀。
            feature_dim: 特征维度（用于 LSTM）。
        """
        import joblib

        # 加载 LogisticRegression
        try:
            self.models["LogisticRegression"] = joblib.load(f"{path}_logisticregression.pkl")
            self.models["LogisticRegression"].is_fitted = True
        except FileNotFoundError:
            print(f"警告: 未找到 LogisticRegression 模型文件")

        # 加载 LSTM
        try:
            import torch
            checkpoint = torch.load(f"{path}_lstm.pkl", map_location=DEVICE)
            lstm_model = LSTMModel(
                input_dim=checkpoint["config"]["input_dim"],
                hidden_dim=checkpoint["config"]["hidden_dim"],
                num_layers=checkpoint["config"]["num_layers"],
                dropout=checkpoint["config"]["dropout"],
                sequence_length=checkpoint["config"]["sequence_length"],
            )
            lstm_model.model = lstm_model._build_model().to(DEVICE)
            lstm_model.model.load_state_dict(checkpoint["model_state_dict"])
            lstm_model.scaler = checkpoint["scaler"]
            lstm_model.is_fitted = True
            self.models["LSTM"] = lstm_model
        except FileNotFoundError:
            print(f"警告: 未找到 LSTM 模型文件")

        # 加载 LightGBM
        try:
            self.models["LightGBM"] = joblib.load(f"{path}_lightgbm.pkl")
            self.models["LightGBM"].is_fitted = True
        except FileNotFoundError:
            print(f"警告: 未找到 LightGBM 模型文件")

        print(f"模型加载完成")


if __name__ == "__main__":
    # 示例用法
    from src.models.ml_track.features import FeatureEngineer

    print("=" * 60)
    print("  ML Track 示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=500, freq="B")
    base_price = 100
    returns = np.random.randn(500) * 0.02
    prices = base_price * (1 + returns).cumprod()

    sample_df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(500) * 0.005),
        "high": prices * (1 + np.abs(np.random.randn(500)) * 0.01),
        "low": prices * (1 - np.abs(np.random.randn(500)) * 0.01),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, 500),
    }, index=dates)

    # 确保价格合理性
    sample_df["high"] = sample_df[["open", "high", "close"]].max(axis=1)
    sample_df["low"] = sample_df[["open", "low", "close"]].min(axis=1)

    print(f"\n样本数据形状: {sample_df.shape}")

    # 计算特征
    print("\n计算技术因子...")
    engineer = FeatureEngineer()
    features_df = engineer.compute_all_features(sample_df, drop_na=True)
    features_df = engineer.create_target(features_df, forward_period=1)

    # 删除最后一天（没有目标值）
    features_df = features_df.dropna()

    print(f"特征数据形状: {features_df.shape}")
    print(f"特征数量: {len(engineer.feature_names)}")

    # 训练模型
    portfolio = MLStrategyPortfolio(
        lstm_epochs=20,
        lstm_sequence_length=10,
    )
    portfolio.fit(features_df, test_size=0.2)

    # 生成信号
    print("\n生成预测信号...")
    signals_df = portfolio.predict(features_df.tail(10), symbol="SAMPLE")
    print(f"\n信号 DataFrame:")
    print(signals_df.to_string())

    # 特征重要性
    print("\n特征重要性 (Top 10):")
    importance_df = portfolio.get_feature_importance()
    if not importance_df.empty:
        print(importance_df.head(10).to_string())