from loguru import logger
import tensorboard

logger.add("logs/train_{time}.log")