import cv2
import time

from os.path import join
from collections import defaultdict

from processor import FrameProcessor
from visualizer import EventVisualizer
from extractor import EventExtractor
from sender import EventSender
from utils import async_enumerate, create_logger


class EventWatcher:
    logger = create_logger(__name__)

    def __init__(self, config):
        self.config = config
        self.reader, self.writer = None, None

    async def watch_events(self, file_name):
        input_path = str(join(self.config.inputs_root, file_name))
        output_path = str(join(self.config.outputs_root, file_name))

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
        extractor = EventExtractor(self.config.extractor, input_path, total_frames, frame_size, line_data, fourcc, fps)
        sender = EventSender(self.config.sender, input_path)

        total_stats = defaultdict(lambda: 0)

        await extractor.split_frames()

        async for index, frame in async_enumerate(self._read_frame()):
            if self.config.frames_skip and index % (self.config.frames_skip + 1):
                continue

            self.logger.debug("Processing frame {} of {}...".format(index, total_frames))
            start = time.time()

            tracks = await processor.get_tracks(frame, self.config.filter_label)

            stop = time.time()
            self.logger.debug("Processed in {} sec.".format(round(stop - start, 4)))

            extractor.snapshot_tracks(index, tracks)

            events, stats = processor.get_events(tracks)
            sender.send_events(events)

            for event in events:
                # TODO: Num constants for events instead of str literals.
                # TODO: Refactor events content, format and DB fields.
                if event['event_name'] == 'line intersection':
                    extractor.register_event(index, event['track_id'])

            if extractor.is_ready(index):
                extractor.extract_event()

            for key in stats:
                total_stats[key] += stats[key]

            frame = visualizer.draw_annotations(frame, tracks)

            self.writer.write(frame)

        self.logger.info("Detected {} new objects and {} line intersections.".format(*total_stats.values()))

        self.reader.release()
        self.writer.release()

    async def _read_frame(self):
        while self.reader.isOpened():
            ret, frame = self.reader.read()

            if ret:
                yield frame
            else:
                break
