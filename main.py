# TODO: Optimize configs structure.


from os.path import join
from omegaconf import OmegaConf

from fastapi import FastAPI, status, File, UploadFile
from fastapi.staticfiles import StaticFiles

from watcher import EventWatcher
from saver import EventSaver


inputs = StaticFiles(directory='inputs')
outputs = StaticFiles(directory='outputs')
configs = StaticFiles(directory='configs')

app = FastAPI()
app.mount('/inputs', inputs, name='inputs')
app.mount('/outputs', outputs, name='outputs')
app.mount('/configs', configs, name='configs')


@app.post('/process', status_code=status.HTTP_204_NO_CONTENT)
async def process_video(config_object: UploadFile = File(...), video_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    video_path = join(inputs.directory, video_object.filename)

    config = OmegaConf.load(config_path)

    watcher = EventWatcher(config.watcher)
    await watcher.watch_events(video_path)


@app.post('/save', status_code=status.HTTP_201_CREATED)
async def save_events(config_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    config = OmegaConf.load(config_path)

    saver = EventSaver(config.saver)
    await saver.save_events()
