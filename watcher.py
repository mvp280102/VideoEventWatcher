import cv2
import time
import numpy as np
import pandas as pd

from os import mkdir, listdir
from os.path import join, splitext, exists

from ffmpeg.asyncio import FFmpeg

from utils import async_enumerate, create_logger
from constants import NEW_OBJECT, LINE_INTERSECTION


class EventWatcher:
    logger = create_logger(__name__)

    def __init__(self, config, processor, visualizer, sender):
        self.config = config
        self.writer = None
        self.frames_dir, self.tracks_dir = None, None

        self.processor = processor
        self.visualizer = visualizer
        self.sender = sender

        self.columns = ['frame_index', 'x_min', 'y_min', 'x_max', 'y_max', 'track_id']
        self.total_tracks = pd.DataFrame(columns=self.columns)

    async def watch_events(self, file_name):
        input_path = str(join(self.config.inputs_root, file_name))
        output_path = str(join(self.config.outputs_root, file_name))

        base_name = splitext(file_name)[0]

        self.frames_dir = str(join(self.config.frames_root, base_name))
        self.tracks_dir = str(join(self.config.tracks_root, base_name))

        if not exists(self.frames_dir):
            mkdir(self.frames_dir)

        if not exists(self.tracks_dir):
            mkdir(self.tracks_dir)

        fourcc = cv2.VideoWriter.fourcc(*self.config.fourcc)
        fps = self.config.fps

        total_stats = dict.fromkeys([NEW_OBJECT, LINE_INTERSECTION], 0)

        await self._split_frames(input_path, self.frames_dir)

        total_frames = len(listdir(self.frames_dir))

        # crutch for getting frame size:
        temp_frame = cv2.imread(join(self.frames_dir, listdir(self.frames_dir)[0]))
        frame_height, frame_width, _ = temp_frame.shape
        frame_size = (frame_width, frame_height)

        self.writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=frame_size)

        async for index, frame in async_enumerate(self._read_frame()):
            if self.config.frames_skip and index % (self.config.frames_skip + 1):
                continue

            self.logger.debug("Processing frame {} of {}...".format(index, total_frames))
            start = time.time()

            raw_tracks = await self.processor.get_tracks(frame, self.config.target_labels)

            stop = time.time()
            self.logger.debug("Processed in {} sec.".format(round(stop - start, 4)))

            index_tracks = np.insert(raw_tracks, 0, index + 1, 1)
            index_tracks_df = pd.DataFrame(index_tracks, columns=self.columns)

            # increment index to match with ffmpeg frames output:
            tracks_path = join(self.tracks_dir, '{}.csv'.format(index + 1))
            index_tracks_df.to_csv(tracks_path, index=False)

            self.logger.info("Dump tracks for frame {} of '{}' video to '{}' file."
                             .format(index + 1, input_path, tracks_path))

            raw_events, raw_stats = self.processor.get_events(index_tracks)

            event_data = {'video_path': input_path}
            events = [{**event, **event_data} for event in raw_events]

            self.sender.send_events(events)

            for key in raw_stats:
                total_stats[key] += raw_stats[key]

            frame = self.visualizer.draw_annotations(frame, raw_tracks)

            self.writer.write(frame)

        self.logger.info("Detected {} new objects and {} line intersections.".format(*total_stats.values()))

        self.writer.release()

    async def _read_frame(self):
        for frame_name in sorted(listdir(self.frames_dir), key=lambda name: int(splitext(name)[0])):
            yield cv2.imread(join(self.frames_dir, frame_name))

    async def _split_frames(self, input_path, frames_dir):
        ffmpeg = FFmpeg().option('y').input(input_path).output(join(frames_dir, '%d.png'))
        await ffmpeg.execute()

        self.logger.info("Split '{}' video into frames to '{}' dir.".format(input_path, frames_dir))
