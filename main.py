from os.path import join
from omegaconf import OmegaConf

from fastapi import FastAPI, status, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from watcher import EventWatcher
from receiver import EventReceiver
from saver import EventSaver


configs = StaticFiles(directory='configs')

app = FastAPI()
app.mount('/configs', configs, name='configs')


@app.get('/', response_class=RedirectResponse)
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post('/process', status_code=status.HTTP_204_NO_CONTENT)
async def process_video(config_object: UploadFile = File(...), video_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    config = OmegaConf.load(config_path)

    watcher = EventWatcher(config.watcher)
    await watcher.watch_events(video_object.filename)


@app.post('/save', status_code=status.HTTP_201_CREATED)
async def save_events(config_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    config = OmegaConf.load(config_path)

    saver = EventSaver(config.saver)
    receiver = EventReceiver(config.receiver)

    await receiver.receive_events([saver.save_event])
