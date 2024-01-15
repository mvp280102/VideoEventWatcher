import cv2
import pandas as pd

from os import mkdir, listdir
from os.path import join, basename, splitext, exists

from utils import create_logger


class EventExtractor:
    logger = create_logger(__name__)

    def __init__(self, config, visualizer):
        self.frames_root = config.frames_root
        self.tracks_root = config.tracks_root
        self.events_root = config.events_root

        self.format = config.format
        self.fourcc = config.fourcc
        self.fps = config.fps

        self.frames_before = config.sec_before * self.fps
        self.frames_after = config.sec_after * self.fps

        self.visualizer = visualizer

    async def extract_event(self, event):
        cur_frame_index = event['frame_index']
        track_id = event['track_id']

        base_name = splitext(basename(event['video_path']))[0]

        frames_dir = str(join(self.frames_root, base_name))
        tracks_dir = str(join(self.tracks_root, base_name))
        events_dir = str(join(self.events_root, base_name))

        total_frames = len(listdir(frames_dir))

        if not exists(events_dir):
            mkdir(events_dir)

        cur_frame = cv2.imread(join(frames_dir, '{}.png'.format(cur_frame_index)))
        frame_height, frame_width, _ = cur_frame.shape
        frame_size = (frame_width, frame_height)

        cur_tracks = pd.read_csv(join(tracks_dir, '{}.csv'.format(cur_frame_index)))
        columns = cur_tracks.columns
        tracks_df = pd.DataFrame(columns=columns, index=None)

        for tracks_file_name in listdir(tracks_dir):
            tracks = pd.read_csv(join(tracks_dir, tracks_file_name))
            tracks_df = pd.concat([tracks_df, pd.DataFrame(tracks, columns=columns)], ignore_index=True)

        req_tracks_df = tracks_df.loc[(tracks_df['track_id'] == track_id)]
        start_frame_index = max(1, req_tracks_df['frame_index'].min() - self.frames_before)
        end_frame_index = min(total_frames, req_tracks_df['frame_index'].max() + self.frames_after)

        output_path = join(events_dir, '{}_{}.{}'.format(track_id, cur_frame_index, self.format))
        fourcc = cv2.VideoWriter.fourcc(*self.fourcc)
        writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=self.fps, frameSize=frame_size)

        for index in range(start_frame_index, end_frame_index):
            track = req_tracks_df.loc[(tracks_df['frame_index'] == index)]
            frame = cv2.imread(join(frames_dir, '{}.png'.format(index)))
            frame = self.visualizer.draw_annotations(frame, track.to_numpy()[:, 1:])

            writer.write(frame)

        writer.release()

        self.logger.info("Write extracted event with name '{}' to {} file.".format(event['event_name'], output_path))
