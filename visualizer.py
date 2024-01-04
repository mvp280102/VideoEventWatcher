import cv2

from utils import get_line_coefficients, random_color


# TODO: Config for visualizer (bbs and line params).
class EventVisualizer:
    def __init__(self, frame_size, line_data):
        self.frame_width, self.frame_height = frame_size
        self.line_data = line_data
        self._track_colors = {}

    def draw_annotations(self, frame, tracks):
        if self.line_data:
            frame = self._draw_line(frame, *get_line_coefficients(*self.line_data))

        for track in tracks:
            frame = self._draw_bounding_box(frame, track)

        return frame

    def _draw_bounding_box(self, frame, track):
        x_min, y_min, x_max, y_max, track_id = track
        x_anchor, y_anchor = int((x_min + x_max) / 2), int(y_max)

        if track_id not in self._track_colors:
            self._track_colors[track_id] = random_color()

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1
        color = self._track_colors[track_id]
        thickness = 2
        radius = -1

        factor = round(max((x_max - x_min), (y_max - y_min)) / 64) + 1

        frame = cv2.circle(frame, (int(x_anchor), y_anchor), thickness * factor, color, radius)
        frame = cv2.rectangle(frame, (x_max, y_max), (x_min, y_min), color, thickness)
        frame = cv2.putText(frame, f'{track_id}', (x_min, y_min - 10), font, scale, color, thickness)

        return frame

    def _draw_line(self, frame, line_k, line_b):
        x1 = -line_b / line_k
        y1 = line_b

        point1 = (int(x1), 0) if x1 > 0 else (0, int(y1))

        x2 = (self.frame_height - line_b) / line_k
        y2 = line_k * self.frame_width + line_b

        point2 = (int(x2), self.frame_height) if x2 > 0 else (self.frame_width, int(y2))

        color = (255, 255, 255)
        thickness = 4

        frame = cv2.line(frame, point1, point2, color, thickness)

        return frame
