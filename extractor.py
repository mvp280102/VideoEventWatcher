import cv2
import numpy as np
import pandas as pd

from os import mkdir
from collections import deque
from os.path import join, basename, splitext, exists

from ffmpeg.asyncio import FFmpeg

from visualizer import EventVisualizer
from utils import create_logger
from constants import SEC_BEFORE_EVENT, SEC_AFTER_EVENT


class EventExtractor:
    logger = create_logger(__name__)

    def __init__(self, config, input_path, total_frames, frame_size, line_data, fourcc, fps):
        self.input_path = input_path

        self.total_frames = total_frames
        self.frame_size = frame_size
        self.line_data = line_data
        self.fourcc = fourcc
        self.fps = fps

        base_name = splitext(basename(self.input_path))[0]

        self.frames_dir = str(join(config.frames_root, base_name))
        self.events_dir = str(join(config.events_root, base_name))
        self.tracks_path = str(join(config.tracks_root, base_name + '.csv'))

        if not exists(self.frames_dir):
            mkdir(self.frames_dir)

        if not exists(self.events_dir):
            mkdir(self.events_dir)

        self.columns = ['frame_index', 'x_min', 'y_min', 'x_max', 'y_max', 'track_id']
        self.tracks = pd.DataFrame(columns=self.columns)

        self.event_queue = deque()

    async def split_frames(self):
        self.logger.info("Split '{}' video into frames to '{}' dir.".format(self.input_path, self.frames_dir))
        ffmpeg = FFmpeg().option('y').input(self.input_path).output(join(self.frames_dir, '%d.png'))
        await ffmpeg.execute()

    def snapshot_tracks(self, index, tracks):
        for track in tracks:
            track = np.insert(track, 0, index)
            self.tracks = pd.concat([self.tracks, pd.DataFrame([track], columns=self.columns)], ignore_index=True)

    def save_tracks(self):
        self.logger.info("Save tracks for '{}' video to '{}' file.".format(self.input_path, self.tracks_path))
        self.tracks.to_csv(self.tracks_path)

    def register_event(self, index, track_id):
        self.event_queue.append((index, track_id))

    def is_ready(self, index):
        return self.event_queue and index >= self.event_queue[0][0] * SEC_AFTER_EVENT

    def extract_event(self):
        frame_index, track_id = self.event_queue.popleft()

        output_path = join(self.events_dir, '{}_{}.avi'.format(frame_index, track_id))
        visualizer = EventVisualizer(self.frame_size, self.line_data)
        writer = cv2.VideoWriter(filename=output_path, fourcc=self.fourcc, fps=self.fps, frameSize=self.frame_size)

        start_index = max(1, frame_index - SEC_BEFORE_EVENT * self.fps)
        end_index = min(self.total_frames, frame_index + SEC_AFTER_EVENT * self.fps)

        for index in range(start_index, end_index):
            track = self.tracks.loc[(self.tracks['frame_index'] == index) & (self.tracks['track_id'] == track_id)]

            frame = cv2.imread(join(self.frames_dir, '{}.png'.format(index)))
            frame = visualizer.draw_annotations(frame, track.to_numpy()[:, 1:])

            writer.write(frame)

        writer.release()
