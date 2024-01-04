import cv2
import time

from os.path import basename
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
        self.total_frames, self.fps = None, None
        self.frame_width, self.frame_height = None, None

    async def watch_events(self, input_path):
        self.reader = cv2.VideoCapture(input_path)

        self.total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        self.fps = int(self.reader.get(cv2.CAP_PROP_FPS))

        self.frame_width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = input_path.replace('inputs', 'outputs')
        tracks_path = input_path.replace('inputs', 'tracks').replace('.avi', '.csv')

        fourcc = cv2.VideoWriter.fourcc(*'XVID')

        self.writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=self.fps,
                                      frameSize=(self.frame_width, self.frame_height))

        line_data = self.config.line_data if 'line_data' in self.config else None

        processor = FrameProcessor(self.config.processor, (self.frame_width, self.frame_height), line_data)
        visualizer = EventVisualizer((self.frame_width, self.frame_height), line_data)
        extractor = EventExtractor(tracks_path)
        sender = EventSender(basename(input_path), self.config.sender.queue_name, self.config.sender.host_name)

        total_stats = defaultdict(lambda: 0)

        async for index, frame in async_enumerate(self._read_frame()):
            if self.config.frames_skip and index % (self.config.frames_skip + 1):
                continue

            self.logger.debug("Processing frame {} of {}...".format(index, self.total_frames))
            start = time.time()

            tracks = await processor.get_tracks(frame, self.config.filter_label)

            stop = time.time()
            self.logger.debug("Processed in {} sec.".format(round(stop - start, 4)))

            extractor.write_tracks(index, tracks)

            events, stats = processor.get_events(tracks)
            sender.send_events(events)

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
