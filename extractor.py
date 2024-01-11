import cv2
import pandas as pd

from os import mkdir, listdir
from os.path import join, basename, splitext, exists

from utils import create_logger


# TODO: Rewrite this junk in a normal way with several methods.
class EventExtractor:
    logger = create_logger(__name__)

    def __init__(self, config, visualizer):
        self.frames_root = config.frames_root
        self.tracks_root = config.tracks_root
        self.events_root = config.events_root

        self.req_event_name = config.event_name

        self.sec_before_event = config.sec_before_event
        self.sec_after_event = config.sec_after_event

        self.format = config.format
        self.fourcc = config.fourcc
        self.fps = config.fps

        self.visualizer = visualizer

    async def extract_event(self, event):
        event_name = event['event_name']

        if event_name != self.req_event_name:
            self.logger.debug("Skip inappropriate event with name '{}'.".format(event_name))
            return

        cur_frame_index = event['frame_index']
        track_id = event['track_id']

        base_name = splitext(basename(event['video_path']))[0]

        frames_dir = str(join(self.frames_root, base_name))
        tracks_dir = str(join(self.tracks_root, base_name))
        events_dir = str(join(self.events_root, base_name))

        if not exists(events_dir):
            mkdir(events_dir)

        total_frames = len(listdir(frames_dir))

        start_frame_index = max(1, cur_frame_index - self.sec_before_event * self.fps)
        end_frame_index = min(total_frames, cur_frame_index + self.sec_after_event * self.fps)

        # TODO: Something with last two frames.
        if not exists(join(tracks_dir, '{}.csv'.format(end_frame_index))):
            self.logger.info("Skip event with name {} at frame {} due to lack of processed frames."
                             .format(event_name, cur_frame_index))
            return

        cur_frame = cv2.imread(join(frames_dir, '{}.png'.format(cur_frame_index)))
        frame_height, frame_width, _ = cur_frame.shape
        frame_size = (frame_width, frame_height)

        cur_tracks = pd.read_csv(join(tracks_dir, '{}.csv'.format(cur_frame_index)))
        columns = cur_tracks.columns.append(pd.Index(['frame_index']))
        tracks_df = pd.DataFrame(columns=columns, index=None)

        for tracks_file_name in listdir(tracks_dir):
            tracks = pd.read_csv(join(tracks_dir, tracks_file_name))
            tracks.insert(0, 'frame_index', int(splitext(tracks_file_name)[0]))
            tracks_df = pd.concat([tracks_df, pd.DataFrame(tracks, columns=columns)], ignore_index=True)

        output_path = join(events_dir, '{}_{}.{}'.format(cur_frame_index, track_id, self.format))
        fourcc = cv2.VideoWriter.fourcc(*self.fourcc)
        writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=self.fps, frameSize=frame_size)

        for index in range(start_frame_index, end_frame_index):
            track = tracks_df.loc[(tracks_df['frame_index'] == index) & (tracks_df['track_id'] == track_id)]
            frame = cv2.imread(join(frames_dir, '{}.png'.format(index)))
            frame = self.visualizer.draw_annotations(frame, track.to_numpy()[:, :-1])

            writer.write(frame)

        writer.release()
