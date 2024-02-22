import torch
import asyncio
import numpy as np
import pandas as pd

from os import mkdir
from datetime import datetime
from argparse import Namespace
from os.path import exists, join, splitext

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from constants import NEW_OBJECT, LINE_INTERSECTION, TRACKS_DF_COLUMNS, DATETIME_FMT, INTERSECT_THRESH, DELAY_TIME
from utils import create_logger, get_line_coefficients


class FrameProcessor:
    logger = create_logger(__name__)

    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint_path = str(join(config.ckpt_root, config.detector_name + '.pt'))

        self.input_size = config.input_size
        self.tracks_root = config.tracks_root

        self.detector = YOLO(model=checkpoint_path, verbose=False)
        self.detector.to(self.device)

        tracker_args = Namespace(track_buffer=config.track_buffer,
                                 match_thresh=config.match_thresh,
                                 new_track_thresh=config.new_track_thresh,
                                 track_low_thresh=config.track_low_thresh,
                                 track_high_thresh=config.track_high_thresh)

        self.tracker = BYTETracker(tracker_args)
        self.total_tracks = set()
        self.last_tracks = []

        self.frames_skip = config.frames_skip
        self.target_labels = config.target_labels

        self.line_angle = config.line_angle
        self.line_point = config.line_point

    async def process_frame(self, file_name, index, frame):
        base_name = splitext(file_name)[0]
        tracks_dir = str(join(self.tracks_root, base_name))

        if not exists(tracks_dir):
            mkdir(tracks_dir)

        tracks = await self._get_tracks(index, frame)
        self._dump_tracks(tracks, tracks_dir)
        events = self._find_events(file_name, tracks)

        return events

    async def _get_tracks(self, index, frame):
        if index % (self.frames_skip + 1):
            result_tracks = self.last_tracks
        else:
            with torch.no_grad():
                result = self.detector(source=frame, classes=self.target_labels, imgsz=self.input_size)[0]
                detections = result.boxes.cpu()

            # to release GIL:
            await asyncio.sleep(DELAY_TIME)

            result_tracks = self.tracker.update(detections.numpy())

            # to get around tracker mysterious bug:
            if result_tracks.any():
                self.last_tracks = result_tracks
            else:
                result_tracks = self.last_tracks

        result_tracks = np.insert(result_tracks, 0, index + 1, 1)
        result_tracks = result_tracks[:, :6].astype('int')

        return result_tracks

    def _find_events(self, file_name, tracks):
        line_k, line_b = None, None

        event_keys = ('timestamp', 'file_name', 'frame_index', 'track_id', 'event_name')

        event_data = []

        if self.line_angle and self.line_point:
            line_k, line_b = get_line_coefficients(self.line_angle, self.line_point)

        for track in tracks:
            index, x_min, y_min, x_max, y_max, track_id = track
            x_pos, y_pos = int((x_min + x_max) / 2), int(y_max)
            timestamp = datetime.now().strftime(DATETIME_FMT)

            if track_id not in self.total_tracks:
                self.total_tracks.add(track_id)

                event_name = NEW_OBJECT
                event_values = (timestamp, file_name, int(index), int(track_id), event_name)
                event_data.append(dict(zip(event_keys, event_values)))

            if self.line_angle and self.line_point and abs(line_k * x_pos + line_b - y_pos) < INTERSECT_THRESH:
                event_name = LINE_INTERSECTION
                event_values = (timestamp, file_name, int(index), int(track_id), event_name)
                event_data.append(dict(zip(event_keys, event_values)))

        return event_data

    def _filter_detections(self, detections):
        if not self.target_labels:
            return detections

        result = []

        for label in self.target_labels:
            filtered_detections = [detection for detection in filter(lambda det: int(det[-1]) == label, detections)]

            if filtered_detections:
                result.append(torch.stack(filtered_detections))

        return torch.cat(result)

    @staticmethod
    def _dump_tracks(tracks, tracks_dir):
        tracks_df = pd.DataFrame(tracks, columns=TRACKS_DF_COLUMNS)
        frame_index = tracks_df['frame_index'][0]

        tracks_path = join(tracks_dir, '{}.csv'.format(frame_index))
        tracks_df.to_csv(tracks_path, index=False)
