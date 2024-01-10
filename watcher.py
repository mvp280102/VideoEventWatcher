import cv2
import time
import numpy as np
import pandas as pd

from os import mkdir
from collections import defaultdict
from os.path import join, splitext, exists

from ffmpeg.asyncio import FFmpeg

from processor import FrameProcessor
from visualizer import EventVisualizer
from sender import EventSender
from utils import async_enumerate, create_logger


class EventWatcher:
    logger = create_logger(__name__)

    def __init__(self, config):
        self.config = config
        self.reader, self.writer = None, None

        self.columns = ['x_min', 'y_min', 'x_max', 'y_max', 'track_id']
        self.total_tracks = pd.DataFrame(columns=self.columns)

    async def watch_events(self, file_name):
        input_path = str(join(self.config.inputs_root, file_name))
        output_path = str(join(self.config.outputs_root, file_name))

        base_name = splitext(file_name)[0]

        frames_dir = str(join(self.config.frames_root, base_name))
        tracks_dir = str(join(self.config.tracks_root, base_name))

        if not exists(frames_dir):
            mkdir(frames_dir)

        if not exists(tracks_dir):
            mkdir(tracks_dir)

        self.reader = cv2.VideoCapture(input_path)

        total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        fps = int(self.reader.get(cv2.CAP_PROP_FPS))

        frame_width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

        fourcc = cv2.VideoWriter.fourcc(*'XVID')

        self.writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=frame_size)

        line_data = self.config.line_data if 'line_data' in self.config else None

        processor = FrameProcessor(self.config.processor, frame_size, line_data)
        visualizer = EventVisualizer(frame_size, line_data)
        sender = EventSender(self.config.sender)

        total_stats = defaultdict(lambda: 0)

        await self._split_frames(input_path, frames_dir)

        async for index, frame in async_enumerate(self._read_frame()):
            if self.config.frames_skip and index % (self.config.frames_skip + 1):
                continue

            self.logger.debug("Processing frame {} of {}...".format(index, total_frames))
            start = time.time()

            raw_tracks = await processor.get_tracks(frame, self.config.filter_label)

            stop = time.time()
            self.logger.debug("Processed in {} sec.".format(round(stop - start, 4)))

            raw_tracks_df = pd.DataFrame(raw_tracks, columns=self.columns)
            index_tracks = np.insert(raw_tracks, 0, index, 1)

            tracks_path = join(tracks_dir, '{}.csv'.format(index))
            raw_tracks_df.to_csv(tracks_path, index=False)

            self.logger.info("Save tracks for frame {} of '{}' video to '{}' file."
                             .format(index, input_path, tracks_path))

            raw_events, raw_stats = processor.get_events(index_tracks)

            event_data = {'video_path': input_path}
            events = [{**event, **event_data} for event in raw_events]

            sender.send_events(events)

            for key in raw_stats:
                total_stats[key] += raw_stats[key]

            frame = visualizer.draw_annotations(frame, raw_tracks)

            self.writer.write(frame)

        self.logger.info("Detected {} new objects and {} line intersections."
                         .format(*total_stats.values()))

        self.reader.release()
        self.writer.release()

    async def _read_frame(self):
        while self.reader.isOpened():
            ret, frame = self.reader.read()

            if ret:
                yield frame
            else:
                break

    async def _split_frames(self, input_path, frames_dir):
        ffmpeg = FFmpeg().option('y').input(input_path).output(join(frames_dir, '%d.png'))
        await ffmpeg.execute()

        self.logger.info("Split '{}' video into frames to '{}' dir.".format(input_path, frames_dir))
