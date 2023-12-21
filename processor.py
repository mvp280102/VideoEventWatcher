import cv2
import torch

from os.path import join
from datetime import datetime

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

from bytetracker import BYTETracker

from utils import create_logger, get_line_coefficients, filter_detections, random_color


class VideoProcessor:

    # TODO: Add line drawing.
    # TODO: Add inference time.

    def __init__(self, model_name, checkpoints_dir, input_size=None, frames_skip=None, filter_label=0):
        self.logger = create_logger(__name__)

        self.reader, self.writer = None, None
        self.total_frames, self.fps = None, None
        self.frame_width, self.frame_height = None, None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint_path = join(checkpoints_dir, model_name + '.pth')
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        detector_exp = get_exp(exp_name=model_name)

        self.input_size = input_size if input_size else detector_exp.input_size
        self.num_classes = detector_exp.num_classes
        self.conf_thresh = detector_exp.test_conf
        self.nms_thresh = detector_exp.nmsthre

        self.detector_model = detector_exp.get_model()
        self.detector_model.cuda() if self.device == 'cuda' else self.detector_model.cpu()
        self.detector_model.eval()
        self.detector_model.load_state_dict(checkpoint['model'])

        self.tracker_model = BYTETracker()
        self.track_colors = {}
        self._events = []

        self.frames_skip = frames_skip
        self.filter_label = filter_label

    def process_video(self, input_path):
        self.reader = cv2.VideoCapture(input_path)

        self.total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        self.fps = int(self.reader.get(cv2.CAP_PROP_FPS))

        self.frame_width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = input_path.replace('inputs', 'outputs')
        fourcc = cv2.VideoWriter.fourcc(*'XVID')

        self.writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=self.fps,
                                      frameSize=(self.frame_width, self.frame_height))

        for index, raw_frame in enumerate(self._read_frame()):
            if self.frames_skip and index % (self.frames_skip + 1):
                continue

            # self.logger.debug(f'Processing frame {index} of {self.total_frames}...')
            processed_frame = self._process_frame(raw_frame, self.filter_label)
            self.writer.write(processed_frame)

        self.reader.release()
        self.writer.release()

    def _read_frame(self):
        while self.reader.isOpened():
            ret, frame = self.reader.read()

            if ret:
                yield frame
            else:
                break

    def _process_frame(self, frame, label):
        input_frame = self._prepare_frame(frame)

        with torch.no_grad():
            raw_detections = self.detector_model(input_frame)

        raw_detections = postprocess(raw_detections, self.num_classes, self.conf_thresh, self.nms_thresh, True)
        filtered_detections = filter_detections(raw_detections[0], label).cpu()

        ratio = min(self.input_size[1] / self.frame_width, self.input_size[0] / self.frame_height)
        filtered_detections[:, :4] /= ratio

        tracks = self.tracker_model.update(filtered_detections, None)

        line_k, line_b = get_line_coefficients(10, (0, 450))
        frame = self.draw_line(frame, line_k, line_b)

        for track in tracks:
            x_min, y_min, x_max, y_max, track_id = track[:5].astype('int')
            x_anchor, y_anchor = int((x_min + x_max) / 2), y_max

            if track_id not in self.track_colors:
                self.logger.info(f'Event - new object\t\tTrack ID - {track_id}\t\tPosition - ({x_anchor}, {y_anchor})')

                # TODO: Add frame with detected object.

                self.track_colors[track_id] = random_color()
                self._events.append({'timestamp': datetime.now(), 'event_name': 'new object',
                                    'track_id': track_id, 'position': (x_anchor, y_anchor)})

            if abs(line_k * x_anchor + line_b - y_anchor) < 1:
                self.logger.info(f'Event - line intersection\tTrack ID - {track_id}\t\tPosition - ({x_anchor}, {y_anchor})')
                self._events.append({'timestamp': datetime.now(), 'event_name': 'line intersection',
                                    'track_id': track_id, 'position': (x_anchor, y_anchor)})

            # TODO: Add anchor point thickness scaling.

            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 1
            color = self.track_colors[track_id]
            thickness = 2

            frame = cv2.circle(frame, (int(x_anchor), y_anchor), thickness * 2, color, -1)
            frame = cv2.rectangle(frame, (x_max, y_max), (x_min, y_min), color, thickness)
            frame = cv2.putText(frame, f'{track_id}', (x_min, y_min - 10), font, scale, color, thickness)

        return frame

    def _prepare_frame(self, frame):
        frame = ValTransform()(frame, None, self.input_size)[0]
        frame = torch.from_numpy(frame).unsqueeze(0).float()
        frame = frame.cuda() if self.device == 'cuda' else frame.cpu()

        return frame

    def draw_line(self, frame, k, b):
        x1 = -b / k
        y1 = b

        point1 = (int(x1), 0) if x1 > 0 else (0, int(y1))

        x2 = (self.frame_height - b) / k
        y2 = k * self.frame_width + b

        point2 = (int(x2), self.frame_height) if x2 > 0 else (self.frame_width, int(y2))

        color = (255, 255, 255)
        thickness = 4

        return cv2.line(frame, point1, point2, color, thickness)

    @property
    def events(self):
        return self._events
