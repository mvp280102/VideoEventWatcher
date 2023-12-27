from torch import stack
from random import randint
from itertools import count
from math import tan, radians
from datetime import datetime
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter


async def async_enumerate(async_iterable):
    index = count()
    async for item in async_iterable:
        yield next(index), item


# TODO: Params for stream and file handlers.
def create_logger(name, level=INFO):
    logger = getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    stream_handler = StreamHandler()
    file_handler = FileHandler(f'logs/{name}_{datetime.now().strftime("%d.%m.%Y_%H:%M:%S")}.log')
    formatter = Formatter(fmt='%(levelname)s:\t%(name)s\t%(asctime)s\t%(message)s', datefmt='%d.%m.%Y %H:%M:%S')

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_line_coefficients(degree=0, point=(0, 0)):
    # k = tan(alpha), b = y - k * x:
    return tan(radians(degree)), point[1] - tan(radians(degree)) * point[0]


def filter_detections(detections, label):
    return stack([detection for detection in filter(lambda d: int(d[-1]) == label, detections)])


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)
