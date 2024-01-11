from os.path import join
from omegaconf import OmegaConf

from fastapi import FastAPI, status, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from watcher import EventWatcher
from processor import FrameProcessor
from visualizer import EventVisualizer
from sender import EventSender

from receiver import EventReceiver
from extractor import EventExtractor
from saver import EventSaver


configs = StaticFiles(directory='configs')

app = FastAPI()
app.mount('/configs', configs, name='configs')


@app.get('/', response_class=RedirectResponse)
async def redirect_docs():
    return RedirectResponse(url='/docs')


@app.post('/produce', status_code=status.HTTP_204_NO_CONTENT)
async def produce_events(config_object: UploadFile = File(...), video_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    config = OmegaConf.load(config_path)

    processor = FrameProcessor(config.processor)
    visualizer = EventVisualizer(config.processor.line_data)
    sender = EventSender(config.sender)

    watcher = EventWatcher(config.watcher, processor, visualizer, sender)
    await watcher.watch_events(video_object.filename)


@app.post('/consume', status_code=status.HTTP_201_CREATED)
async def consume_events(config_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    config = OmegaConf.load(config_path)

    visualizer = EventVisualizer(config.extractor.line_data)
    extractor = EventExtractor(config.extractor, visualizer)
    saver = EventSaver(config.saver)

    receiver = EventReceiver(config.receiver, extractor, saver)
    await receiver.receive_events()
