import numpy as np
import pandas as pd

from os import mkdir
from os.path import join, basename, splitext, exists

from ffmpeg.asyncio import FFmpeg

from utils import create_logger


class EventExtractor:
    logger = create_logger(__name__)

    def __init__(self, config, input_path):
        self.input_path = input_path

        base_name = splitext(basename(self.input_path))[0]

        self.frames_dir = join(config.frames_root, base_name)
        self.tracks_path = join(config.tracks_root, base_name + '.csv')

        if not exists(self.frames_dir):
            mkdir(self.frames_dir)

        self.columns = ['frame_index', 'x_min', 'y_min', 'x_max', 'y_max', 'track_id']
        self.tracks = pd.DataFrame(columns=self.columns)

    async def split_frames(self):
        self.logger.info("Split '{}' video into frames to '{}' dir.".format(self.input_path, self.frames_dir))
        ffmpeg = FFmpeg().option('y').input(self.input_path).output(join(str(self.frames_dir), 'image%4d.png'))
        await ffmpeg.execute()

    def snapshot_tracks(self, index, tracks):
        for track in tracks:
            track = np.insert(track, 0, index)
            self.tracks = pd.concat([self.tracks, pd.DataFrame([track], columns=self.columns)], ignore_index=True)

    def save_tracks(self):
        self.logger.info("Save tracks for '{}' video to '{}' file.".format(self.input_path, self.tracks_path))
        self.tracks.to_csv(str(self.tracks_path))
