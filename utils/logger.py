import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class Logger:
    def __init__(
        self,
        name: str = "multimodal_classifier",
        log_file: Optional[str] = "training.log",
        level: int = logging.INFO
    ):
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatters
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def debug(self, message: str):
        self.logger.debug(message)

def log_with_timestamp(message: str):
    """Quick logging function with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Create global logger instance
logger = Logger()