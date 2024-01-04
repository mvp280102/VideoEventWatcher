import csv
import numpy

from utils import create_logger


class EventExtractor:
    logger = create_logger(__name__)

    def __init__(self, file_name):
        self.file_name = file_name

    def write_tracks(self, index, tracks):
        self.logger.info("Write tracks for frame {} to '{}' file.".format(index, self.file_name))

        with open(self.file_name, 'a') as file:
            writer = csv.writer(file)

            for track in tracks:
                track = numpy.insert(track, 0, index)
                writer.writerow(track)
