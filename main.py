from omegaconf import DictConfig

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from visualizer import EventVisualizer
from processor import FrameProcessor
from extractor import EventExtractor
from saver import EventSaver

from watcher import EventWatcher

from models import RequestData


app = FastAPI()


@app.get('/', response_class=RedirectResponse)
async def redirect_docs():
    return RedirectResponse(url='/docs')


@app.post('/watch')
async def produce_events(request_data: RequestData):
    config = DictConfig(request_data.config.model_dump())

    visualizer = EventVisualizer(config.processor.line_angle, config.processor.line_point)

    processor = FrameProcessor(config.processor)
    extractor = EventExtractor(config.extractor, visualizer)
    saver = EventSaver(config.saver)

    watcher = EventWatcher(config.watcher, processor, extractor, saver)
    events = await watcher.watch_events(request_data.filename)

    return events
