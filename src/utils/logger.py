import logging
import os


def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(f"{path}/run.log"), exist_ok=True)
    handler = logging.FileHandler(f"{path}/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
