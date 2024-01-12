from random import randint
from itertools import count
from torch import stack, cat
from math import tan, radians
from datetime import datetime
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

from constants import DATETIME_FMT


async def async_enumerate(async_iterable):
    index = count()

    async for item in async_iterable:
        yield next(index), item


def create_logger(name, stream=True, file=True, level=INFO):
    logger = getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = Formatter(fmt='{levelname}:\t{name:<12}{asctime:<24}{message}', datefmt=DATETIME_FMT, style='{')

    if stream:
        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file:
        file_handler = FileHandler(f'logs/{name}_{datetime.now().strftime(DATETIME_FMT.replace(" ", "_"))}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_line_coefficients(degree=0, point=(0, 0)):
    # k = tan(alpha), b = y - k * x:
    return tan(radians(degree)), point[1] - tan(radians(degree)) * point[0]


def filter_detections(detections, labels):
    if not labels:
        return detections

    result = []

    for label in labels:
        filtered_detections = [detection for detection in filter(lambda det: int(det[-1]) == label, detections)]

        if filtered_detections:
            result.append(stack(filtered_detections))

    return cat(result)


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)
