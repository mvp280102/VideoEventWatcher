from pydantic import BaseModel
from typing import List, Optional


class WatcherConfig(BaseModel):
    target_events: List[str]

    duplicate_interval: float
    fps: int

    inputs_root: str
    frames_root: str


class ProcessorConfig(BaseModel):
    detector_name: str
    input_size: int

    ckpt_root: str
    tracks_root: str

    track_buffer: int
    match_thresh: float
    new_track_thresh: float
    track_low_thresh: float
    track_high_thresh: float

    frames_skip: int
    target_labels: List[int]

    line_angle: Optional[int]
    line_point: Optional[List[int]]


class ExtractorConfig(BaseModel):
    outputs_root: str
    frames_root: str
    tracks_root: str
    events_root: str

    sec_before: int
    sec_after: int

    fourcc: str
    fps: int


class SaverConfig(BaseModel):
    sql_dialect: str
    db_driver: str

    db_username: str
    db_password: str

    host_name: str
    db_name: str


class Config(BaseModel):
    watcher: WatcherConfig
    processor: ProcessorConfig
    extractor: ExtractorConfig
    saver: SaverConfig


class RequestData(BaseModel):
    config: Config
    filename: str
