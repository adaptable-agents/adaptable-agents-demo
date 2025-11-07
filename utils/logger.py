"""Logging utilities for the benchmark script."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Global variable to track if logging has been initialized
_logs_dir: Optional[Path] = None


def setup_logging(base_dir: str = "./logs") -> Path:
    """
    Set up logging with a timestamped logs directory.

    Args:
        base_dir: Base directory for logs (default: ./logs)

    Returns:
        Path to the created logs directory
    """
    global _logs_dir

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = Path(base_dir) / timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log file handler
    log_file = logs_dir / "benchmark.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # Get the root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _logs_dir = logs_dir
    return logs_dir


def get_logs_dir() -> Optional[Path]:
    """Get the current logs directory if it has been initialized."""
    return _logs_dir


# Create logger for benchmark
logger = logging.getLogger("benchmark")

