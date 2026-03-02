"""
配置加载模块。

提供统一的配置加载和管理接口，支持从 YAML 文件加载配置。
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """
    配置加载器。

    支持从 YAML 文件加载配置，并提供默认配置。

    Attributes:
        config_dir: 配置文件目录
        configs: 已加载的配置字典
    """

    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器。

        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self.configs: dict[str, Any] = {}

    def load(self, config_name: str) -> dict:
        """
        加载指定配置文件。

        Args:
            config_name: 配置文件名称（不含扩展名）

        Returns:
            配置字典
        """
        if config_name in self.configs:
            return self.configs[config_name]

        config_file = self.config_dir / f"{config_name}.yaml"

        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_file}")
            return {}

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self.configs[config_name] = config
            logger.debug(f"配置已加载: {config_file}")
            return config

        except Exception as e:
            logger.error(f"加载配置文件失败: {config_file} - {e}")
            return {}

    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        获取配置值。

        Args:
            config_name: 配置文件名称
            key: 配置键（支持点号分隔，如 'lstm.params.hidden_dim'）
            default: 默认值

        Returns:
            配置值
        """
        config = self.load(config_name)

        if key is None:
            return config

        # 支持点号分隔的键
        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def reload(self, config_name: Optional[str] = None) -> None:
        """
        重新加载配置。

        Args:
            config_name: 配置文件名称，如果为 None 则重新加载所有
        """
        if config_name:
            if config_name in self.configs:
                del self.configs[config_name]
            self.load(config_name)
        else:
            self.configs.clear()
            for config_file in self.config_dir.glob("*.yaml"):
                self.load(config_file.stem)

    def get_all(self) -> dict[str, Any]:
        """
        获取所有已加载的配置。

        Returns:
            所有配置的字典
        """
        # 加载所有配置文件
        for config_file in self.config_dir.glob("*.yaml"):
            self.load(config_file.stem)

        return self.configs.copy()


# 全局配置加载器实例
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """获取全局配置加载器实例。"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(config_name: str) -> dict:
    """
    快捷函数：加载配置。

    Args:
        config_name: 配置文件名称

    Returns:
        配置字典
    """
    return get_config_loader().load(config_name)


def get_config(config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
    """
    快捷函数：获取配置值。

    Args:
        config_name: 配置文件名称
        key: 配置键
        default: 默认值

    Returns:
        配置值
    """
    return get_config_loader().get(config_name, key, default)


# 预定义的常用配置获取函数
def get_data_config(key: Optional[str] = None, default: Any = None) -> Any:
    """获取数据配置。"""
    return get_config("data_config", key, default)


def get_ml_config(key: Optional[str] = None, default: Any = None) -> Any:
    """获取 ML 配置。"""
    return get_config("ml_config", key, default)


def get_llm_config(key: Optional[str] = None, default: Any = None) -> Any:
    """获取 LLM 配置。"""
    return get_config("llm_config", key, default)


def get_backtest_config(key: Optional[str] = None, default: Any = None) -> Any:
    """获取回测配置。"""
    return get_config("backtest_config", key, default)


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("配置加载器示例")
    print("=" * 60)

    # 加载所有配置
    loader = ConfigLoader()
    all_configs = loader.get_all()

    print(f"\n已加载 {len(all_configs)} 个配置文件:")
    for name in all_configs.keys():
        print(f"  - {name}")

    # 获取特定配置
    print("\nML 配置示例:")
    lstm_hidden = get_ml_config("lstm.architecture.hidden_dim", 64)
    print(f"  LSTM hidden_dim: {lstm_hidden}")

    print("\nLLM 配置示例:")
    ollama_model = get_llm_config("executors.ollama.model", "qwen2.5:7b")
    print(f"  Ollama model: {ollama_model}")

    print("\n回测配置示例:")
    initial_cash = get_backtest_config("backtest.initial_cash", 100000)
    print(f"  Initial cash: {initial_cash}")
