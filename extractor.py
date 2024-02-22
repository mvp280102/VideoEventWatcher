import cv2
import pandas as pd

from collections import deque
from os import mkdir, listdir
from os.path import join, exists, splitext

from constants import TRACKS_DF_COLUMNS
from utils import sorted_listdir, create_logger


class EventExtractor:
    logger = create_logger(__name__)

    def __init__(self, config, visualizer):
        self.outputs_root = config.outputs_root
        self.frames_root = config.frames_root
        self.tracks_root = config.tracks_root
        self.events_root = config.events_root

        self.fourcc = cv2.VideoWriter.fourcc(*config.fourcc)
        self.fps = config.fps

        self.frames_before = config.sec_before * self.fps
        self.frames_after = config.sec_after * self.fps

        self._events_queue = deque()

        self.visualizer = visualizer

    def extract_video(self, file_name):
        base_name = splitext(file_name)[0]

        output_path = str(join(self.outputs_root, file_name))
        frames_dir = str(join(self.frames_root, base_name))
        tracks_dir = str(join(self.tracks_root, base_name))

        # crutch for getting frame size:
        temp_frame = cv2.imread(join(frames_dir, listdir(frames_dir)[0]))
        frame_height, frame_width, _ = temp_frame.shape
        frame_size = (frame_width, frame_height)

        writer = cv2.VideoWriter(filename=output_path, fourcc=self.fourcc, fps=self.fps, frameSize=frame_size)

        for index, (frame, tracks) in enumerate(self._get_frames_tracks(frames_dir, tracks_dir)):
            frame = self.visualizer.draw_annotations(index, frame, tracks)
            writer.write(frame)

        writer.release()

        self.logger.newline()
        self.logger.info("Write totally processed video to '{}' file.".format(output_path))

    def extract_events(self, file_name):
        while self._events_queue:
            events = self._events_queue.popleft()

            frame_index = events[0]['frame_index']

            base_name = splitext(file_name)[0]

            frames_dir = str(join(self.frames_root, base_name))
            tracks_dir = str(join(self.tracks_root, base_name))

            total_frames = len(listdir(frames_dir))

            tracks_df = pd.DataFrame(columns=TRACKS_DF_COLUMNS, index=None)

            for tracks_file_name in listdir(tracks_dir):
                tracks = pd.read_csv(join(tracks_dir, tracks_file_name))
                tracks_df = pd.concat([tracks_df, pd.DataFrame(tracks, columns=TRACKS_DF_COLUMNS)], ignore_index=True)

            for event in events:
                track_id = event['track_id']

                req_track_df = tracks_df.loc[(tracks_df['track_id'] == track_id)]
                start_frame_index = max(1, req_track_df['frame_index'].min() - self.frames_before)
                end_frame_index = min(total_frames, req_track_df['frame_index'].max() + self.frames_after)

                if not exists(str(join(tracks_dir, '{}.csv'.format(end_frame_index)))):
                    self._events_queue.appendleft(events)
                    return
                else:
                    self.logger.newline()
                    self.logger.info("Extracting event(s) for frame {} to video:".format(frame_index))

                    self._extract_event(file_name, event, req_track_df, start_frame_index, end_frame_index)

    def _extract_event(self, file_name, event, req_track_df, start_index, end_index):
        cur_index = event['frame_index']
        track_id = event['track_id']
        event_name = event['event_name']

        base_name, extension = splitext(file_name)

        frames_dir = str(join(self.frames_root, base_name))
        events_dir = str(join(self.events_root, base_name))

        if not exists(events_dir):
            mkdir(events_dir)

        cur_frame = cv2.imread(join(frames_dir, '{}.png'.format(cur_index)))
        frame_height, frame_width, _ = cur_frame.shape
        frame_size = (frame_width, frame_height)

        output_path = join(events_dir, '{}-{}{}'.format(cur_index, track_id, extension))
        writer = cv2.VideoWriter(filename=output_path, fourcc=self.fourcc, fps=self.fps, frameSize=frame_size)

        for index in range(start_index, end_index):
            track = req_track_df.loc[(req_track_df['frame_index'] == index)]
            frame = cv2.imread(join(frames_dir, '{}.png'.format(index)))
            frame = self.visualizer.draw_annotations(index, frame, track)

            writer.write(frame)

        writer.release()

        self.logger.info("Track ID - {}, event name - {}, file - {}.".format(track_id, event_name, output_path))

    @staticmethod
    def _get_frames_tracks(frames_dir, tracks_dir):
        for frame_name, tracks_name in zip(sorted_listdir(frames_dir), sorted_listdir(tracks_dir)):
            frame = cv2.imread(str(join(frames_dir, frame_name)))
            tracks = pd.read_csv(str(join(tracks_dir, tracks_name)))
            yield frame, tracks

    @property
    def events_queue(self):
        return self._events_queue
