import torch
import asyncio

from os.path import join
from datetime import datetime

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

from bytetracker import BYTETracker

from utils import create_logger, get_line_coefficients, filter_detections


class FrameProcessor:
    logger = create_logger(__name__)

    def __init__(self, config, frame_size, line_data):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint_path = join(config.ckpt_dir, config.model_name + '.pth')
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        detector_exp = get_exp(exp_name=config.model_name)

        self.input_size = config.input_size if 'input_size' in config else detector_exp.input_size
        self.num_classes = detector_exp.num_classes
        self.conf_thresh = detector_exp.test_conf
        self.nms_thresh = detector_exp.nmsthre

        self.detector = detector_exp.get_model()
        self.detector.cuda() if self.device == 'cuda' else self.detector.cpu()
        self.detector.eval()
        self.detector.load_state_dict(checkpoint['model'])

        self.tracker = BYTETracker()
        self.total_tracks = set()

        self.frame_width, self.frame_height = frame_size
        self.line_data = line_data

    async def get_tracks(self, frame, label):
        with torch.no_grad():
            raw_detections = self.detector(self._prepare_frame(frame))

        # to release GIL:
        await asyncio.sleep(0.05)

        raw_detections = postprocess(raw_detections, self.num_classes, self.conf_thresh, self.nms_thresh, True)
        filtered_detections = filter_detections(raw_detections[0], label).cpu()

        ratio = min(self.input_size[1] / self.frame_width, self.input_size[0] / self.frame_height)
        filtered_detections[:, :4] /= ratio

        tracks = self.tracker.update(filtered_detections, None)

        return tracks

    # TODO: Refactor event log format.
    def get_events(self, tracks):
        line_k, line_b = None, None
        events = []

        if self.line_data:
            line_k, line_b = get_line_coefficients(*self.line_data)

        for track in tracks:
            x_min, y_min, x_max, y_max, track_id = track[:5].astype('int')
            x_anchor, y_anchor = int((x_min + x_max) / 2), int(y_max)

            if track_id not in self.total_tracks:
                self.logger.info(f'Event - new object\t\tTrack ID - {track_id}\t\tPosition - ({x_anchor}, {y_anchor})')

                self.total_tracks.add(track_id)
                events.append({'timestamp': datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
                               'event_name': 'new object',
                               'track_id': int(track_id),
                               'position': (x_anchor, y_anchor)})

            if self.line_data and abs(line_k * x_anchor + line_b - y_anchor) < 1:
                self.logger.info(f'Event - line intersection\tTrack ID - {track_id}\t\tPosition - ({x_anchor}, {y_anchor})')
                events.append({'timestamp': datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
                               'event_name': 'line intersection',
                               'track_id': int(track_id),
                               'position': (x_anchor, y_anchor)})

        return events

    def _prepare_frame(self, frame):
        frame = ValTransform()(frame, None, self.input_size)[0]
        frame = torch.from_numpy(frame).unsqueeze(0).float()
        frame = frame.cuda() if self.device == 'cuda' else frame.cpu()

        return frame
