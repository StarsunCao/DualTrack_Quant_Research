"""
结构化日志系统模块。

提供统一的日志配置，支持多级别日志输出到不同文件。
使用 loguru 库实现高性能异步日志。
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# 移除默认的日志处理器
logger.remove()

# 日志格式
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


def setup_logger(
    log_dir: Optional[str] = "logs",
    console_level: str = "INFO",
    enable_file_log: bool = True,
    enable_debug_log: bool = False,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    配置结构化日志系统。

    Args:
        log_dir: 日志文件保存目录，默认为 logs/
        console_level: 控制台日志级别 (DEBUG, INFO, WARNING, ERROR)
        enable_file_log: 是否启用文件日志
        enable_debug_log: 是否启用单独的 debug 日志文件
        rotation: 日志文件轮转大小
        retention: 日志文件保留时间
    """
    # 确保日志目录存在
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

    # 添加控制台处理器（带颜色）
    logger.add(
        sys.stdout,
        level=console_level,
        format=CONSOLE_FORMAT,
        colorize=True,
        enqueue=True,  # 异步写入
    )

    if enable_file_log and log_dir:
        # Info 日志（包含 INFO 及以上级别）
        logger.add(
            f"{log_dir}/info.log",
            level="INFO",
            format=FILE_FORMAT,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: record["level"].no >= 20,  # INFO = 20
        )

        # Error 日志（仅 ERROR 及以上级别）
        logger.add(
            f"{log_dir}/error.log",
            level="ERROR",
            format=FILE_FORMAT,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        # Debug 日志（仅当启用时）
        if enable_debug_log:
            logger.add(
                f"{log_dir}/debug.log",
                level="DEBUG",
                format=FILE_FORMAT,
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                enqueue=True,
            )

    logger.info(f"日志系统初始化完成 | 目录: {log_dir} | 控制台级别: {console_level}")


def get_logger(name: Optional[str] = None):
    """
    获取配置好的 logger 实例。

    Args:
        name: 模块名称，用于标识日志来源

    Returns:
        配置好的 logger 实例
    """
    if name:
        return logger.bind(name=name)
    return logger


# 便捷函数
def log_section(title: str, level: str = "INFO") -> None:
    """
    打印带分隔线的章节标题。

    Args:
        title: 标题文本
        level: 日志级别
    """
    line = "=" * 60
    log_func = getattr(logger, level.lower())
    log_func("")
    log_func(line)
    log_func(f"  {title}")
    log_func(line)


def log_phase(phase_num: int, total_phases: int, description: str) -> None:
    """
    打印阶段日志。

    Args:
        phase_num: 当前阶段编号
        total_phases: 总阶段数
        description: 阶段描述
    """
    logger.info(f"[Phase {phase_num}/{total_phases}] {description}")


def log_track(track_name: str, track_num: int, total_tracks: int, message: str = "") -> None:
    """
    打印轨道日志。

    Args:
        track_name: 轨道名称
        track_num: 当前轨道编号
        total_tracks: 总轨道数
        message: 附加消息
    """
    msg = f"【轨道 {track_num}/{total_tracks}】{track_name}"
    if message:
        msg += f" - {message}"
    logger.info(msg)


# 默认初始化（如果没有调用 setup_logger，使用基本配置）
if not logger._core.handlers:  # type: ignore
    logger.add(
        sys.stdout,
        level="INFO",
        format=CONSOLE_FORMAT,
        colorize=True,
    )


if __name__ == "__main__":
    # 示例用法
    setup_logger(log_dir="logs", console_level="DEBUG", enable_debug_log=True)

    log_section("DualTrack Quant Research - 日志系统测试")

    # 测试不同级别
    logger.debug("这是一条 DEBUG 日志")
    logger.info("这是一条 INFO 日志")
    logger.warning("这是一条 WARNING 日志")
    logger.error("这是一条 ERROR 日志")

    # 测试阶段日志
    log_phase(1, 6, "数据获取")
    log_phase(2, 6, "特征工程")

    # 测试轨道日志
    log_track("Logistic Regression", 1, 5, "信号生成中...")
    log_track("LSTM", 2, 5, "训练完成")

    # 测试带异常的日志
    try:
        1 / 0
    except Exception:
        logger.exception("捕获到异常")

    logger.info("日志系统测试完成")
