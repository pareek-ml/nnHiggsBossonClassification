"""
# logger.py
# Logger configuration for the training script.
# """
from loguru import logger

logger.add("logs/train_{time}.log")
