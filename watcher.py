import cv2

from os import mkdir
from os.path import exists, join, splitext

from ffmpeg.asyncio import FFmpeg

from utils import async_enumerate, sorted_listdir, create_logger


class EventWatcher:
    logger = create_logger(__name__)

    def __init__(self, config, processor, extractor, saver):
        self.target_events = config.target_events

        self.duplicate_frames = int(config.duplicate_interval * config.fps)

        self.inputs_root = config.inputs_root
        self.frames_root = config.frames_root

        self.processor = processor
        self.extractor = extractor
        self.saver = saver

        self.unique_events = dict()

    async def watch_events(self, file_name):
        input_path = str(join(self.inputs_root, file_name))

        base_name = splitext(file_name)[0]

        frames_dir = str(join(self.frames_root, base_name))

        if not exists(frames_dir):
            mkdir(frames_dir)

        await self._split_frames(input_path, frames_dir)

        total_events = []

        async for index, frame in async_enumerate(self._get_frames(frames_dir)):
            events = await self.processor.process_frame(file_name, index, frame)
            filtered_events = self._filter_events(events)

            if filtered_events:
                self.extractor.events_queue.append(filtered_events)
                self.saver.save_events(filtered_events)

                total_events.extend(filtered_events)

            self.extractor.extract_events(file_name)

        self.extractor.extract_video(file_name)

        return total_events

    def _filter_events(self, events):
        # filter target events:
        res_events = list(filter(lambda ev: ev['event_name'] in self.target_events, events))

        # remove duplicate events:
        res_events = list(filter(lambda ev: (ev['track_id'], ev['event_name']) not in self.unique_events, res_events))

        for event in res_events:
            frame_index = event['frame_index']
            track_id = event['track_id']
            event_name = event['event_name']

            event_key = (track_id, event_name)

            if event_key not in self.unique_events or \
                    self.unique_events[event_key] - frame_index > self.duplicate_frames:
                self.unique_events[event_key] = frame_index

        return res_events

    async def _split_frames(self, input_path, frames_dir):
        ffmpeg = FFmpeg().option('y').input(input_path).output(join(frames_dir, '%d.png'))
        await ffmpeg.execute()

        self.logger.info("Split '{}' video into frames to '{}' dir.".format(input_path, frames_dir))

    @staticmethod
    async def _get_frames(frames_dir):
        for frame_name in sorted_listdir(frames_dir):
            yield cv2.imread(str(join(frames_dir, frame_name)))
