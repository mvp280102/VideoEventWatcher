import torch
import asyncio

from os.path import join
from datetime import datetime
from collections import Counter

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

from bytetracker import BYTETracker

from utils import create_logger, get_line_coefficients, filter_detections
from constants import NEW_OBJECT, LINE_INTERSECTION, DATETIME_FMT, GIL_DELAY_TIME


class FrameProcessor:
    logger = create_logger(__name__)

    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint_path = str(join(config.ckpt_root, config.model_name + '.pth'))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        detector_exp = get_exp(exp_name=config.model_name)

        self.input_size = config.input_size if 'input_size' in config else detector_exp.input_size
        self.num_classes = detector_exp.num_classes
        self.conf_thresh = detector_exp.test_conf
        self.nms_thresh = detector_exp.nmsthre

        self.detector = detector_exp.get_model()
        self.detector.cuda() if self.device == 'cuda' else self.detector.cpu()
        self.detector.eval()
        self.detector.load_state_dict(checkpoint['model'])

        self.tracker = BYTETracker(track_buffer=config.track_buffer)
        self.total_tracks = set()

        self.line_data = config.line_data if 'line_data' in config else None
        self.intersect_thresh = config.intersect_thresh

    async def get_tracks(self, frame, labels):
        with torch.no_grad():
            raw_detections = self.detector(self._prepare_frame(frame))

        # to release GIL:
        await asyncio.sleep(GIL_DELAY_TIME)

        raw_detections = postprocess(raw_detections, self.num_classes, self.conf_thresh, self.nms_thresh, True)
        filtered_detections = filter_detections(raw_detections[0], labels).cpu()

        frame_height, frame_width, _ = frame.shape

        ratio = min(self.input_size[1] / frame_width, self.input_size[0] / frame_height)
        filtered_detections[:, :4] /= ratio

        tracks = self.tracker.update(filtered_detections, None)

        return tracks[:, :5].astype('int')

    def get_events(self, tracks):
        line_k, line_b = None, None

        event_keys = ('timestamp', 'frame_index', 'track_id', 'event_name')

        event_data = []
        event_names = []

        if self.line_data:
            line_k, line_b = get_line_coefficients(*self.line_data)

        for track in tracks:
            index, x_min, y_min, x_max, y_max, track_id = track
            x_pos, y_pos = int((x_min + x_max) / 2), int(y_max)
            timestamp = datetime.now().strftime(DATETIME_FMT)

            if track_id not in self.total_tracks:
                self.total_tracks.add(track_id)

                event_name = NEW_OBJECT
                event_values = (timestamp, int(index), int(track_id), event_name)
                event_data.append(dict(zip(event_keys, event_values)))
                event_names.append(event_name)

                self.logger.info("New object at frame {} with track ID {} in position ({}, {})."
                                 .format(index, track_id, x_pos, y_pos))

            if self.line_data and abs(line_k * x_pos + line_b - y_pos) < self.intersect_thresh:
                event_name = LINE_INTERSECTION
                event_values = (timestamp, int(index), int(track_id), event_name)
                event_data.append(dict(zip(event_keys, event_values)))
                event_names.append(event_name)

                self.logger.info("Line intersection at frame {} by object with track ID {} in position ({}, {})."
                                 .format(index, track_id, x_pos, y_pos))

        stats = Counter(event_names)

        return event_data, stats

    def _prepare_frame(self, frame):
        frame = ValTransform()(frame, None, self.input_size)[0]
        frame = torch.from_numpy(frame).unsqueeze(0).float()
        frame = frame.cuda() if self.device == 'cuda' else frame.cpu()

        return frame
