from itertools import count
from os import listdir, path
from math import tan, radians
from datetime import datetime
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

from constants import DATETIME_FMT


async def async_enumerate(async_iterable):
    index = count()

    async for item in async_iterable:
        yield next(index), item


def sorted_listdir(target_dir):
    for file_name in sorted(listdir(target_dir), key=lambda name: int(path.splitext(name)[0])):
        yield file_name


def create_logger(name, stream=True, file=True, level=INFO):
    logger = getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    blank_formatter = Formatter(fmt='')
    log_formatter = Formatter(fmt='{levelname}:\t{name:<12}{asctime:<24}{message}', datefmt=DATETIME_FMT, style='{')

    if stream:
        blank_handler = StreamHandler()
        blank_handler.setFormatter(blank_formatter)

        stream_handler = StreamHandler()
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)

        logger.blank_handler = blank_handler
        logger.stream_handler = stream_handler

    if file:
        file_handler = FileHandler(f'logs/{name}_{datetime.now().strftime(DATETIME_FMT.replace(" ", "_"))}.log')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    def _new_line():
        logger.removeHandler(logger.stream_handler)
        logger.addHandler(logger.blank_handler)

        logger.info('')

        logger.removeHandler(logger.blank_handler)
        logger.addHandler(logger.stream_handler)

    logger.newline = _new_line

    return logger


def get_line_coefficients(degree=0, point=(0, 0)):
    # k = tan(alpha), b = y - k * x:
    return tan(radians(degree)), point[1] - tan(radians(degree)) * point[0]
