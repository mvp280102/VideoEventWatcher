# TODO: Remove temporary file.
# TODO: Optimize configs structure.


from os.path import join
from aiofiles import open
from omegaconf import OmegaConf

from fastapi import FastAPI, File, UploadFile

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from watcher import EventWatcher
from saver import EventSaver


inputs = StaticFiles(directory='inputs')
outputs = StaticFiles(directory='outputs')
configs = StaticFiles(directory='configs')

app = FastAPI()
app.mount('/inputs', inputs, name='inputs')
app.mount('/outputs', outputs, name='outputs')
app.mount('/configs', configs, name='configs')


# TODO: Optimize videos and configs location.
@app.post('/upload', response_class=RedirectResponse)
async def upload_data(config_object: UploadFile = File(...), video_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    video_path = join(inputs.directory, video_object.filename)

    async with open(video_path, 'wb') as buffer:
        content = await video_object.read()
        await buffer.write(content)
        await buffer.close()

    return RedirectResponse(url=f'/process?config_path={config_path}&video_path={video_path}')


@app.post('/process', response_class=RedirectResponse)
async def process_video(config_path: str, video_path: str):
    config = OmegaConf.load(config_path)

    watcher = EventWatcher(config.watcher)
    await watcher.watch_events(video_path)

    # return RedirectResponse(url=f'/save?filename={filename}')


# TODO: Remove filename param.
@app.post('/save')
async def save_events(filename: str, config_object: UploadFile = File(...)):
    config_path = join(configs.directory, config_object.filename)
    config = OmegaConf.load(config_path)

    saver = EventSaver(config.saver)
    await saver.save_events(filename)
