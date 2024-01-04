import csv
import numpy

from os import mkdir
from os.path import join, basename, splitext

from ffmpeg.asyncio import FFmpeg

from utils import create_logger


class EventExtractor:
    logger = create_logger(__name__)

    def __init__(self, config, input_path):
        self.config = config

        self.input_path = input_path

        base_name = splitext(basename(self.input_path))[0]
        self.tracks_file_name = join(self.config.tracks_dir, base_name + '.csv')

        self.frames_dir = join(self.config.frames_dir, base_name)
        mkdir(self.frames_dir)

    def write_tracks(self, index, tracks):
        self.logger.info("Write tracks for frame {} to '{}' file.".format(index, self.tracks_file_name))

        with open(self.tracks_file_name, 'a') as file:
            writer = csv.writer(file)

            for track in tracks:
                track = numpy.insert(track, 0, index)
                writer.writerow(track)

    async def split_frames(self):
        self.logger.info("Split '{}' video into frames to '{}' directory.".format(self.input_path, self.frames_dir))

        ffmpeg = (
            FFmpeg()
            .option('y')
            .input(self.input_path)
            .output(
                join(str(self.frames_dir), 'image%4d.png'),
            )
        )

        await ffmpeg.execute()
