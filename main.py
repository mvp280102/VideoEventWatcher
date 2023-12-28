# TODO: Remove temporary file.


from os.path import join
from aiofiles import open

from fastapi import FastAPI, File, UploadFile, HTTPException, status

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# from processor import FrameProcessor
from watcher import EventWatcher
from saver import EventSaver

from credentials import *


inputs = StaticFiles(directory='inputs')
outputs = StaticFiles(directory='outputs')

app = FastAPI()
app.mount('/inputs', inputs, name='inputs')
app.mount('/outputs', outputs, name='outputs')


@app.post('/upload', response_class=RedirectResponse)
async def upload_file(file_object: UploadFile = File(...)):
    file_path = join(inputs.directory, file_object.filename)

    async with open(file_path, 'wb') as buffer:
        content = await file_object.read()
        await buffer.write(content)
        await buffer.close()

    return RedirectResponse(url=f'/process?filename={file_object.filename}')


@app.post('/process', response_class=RedirectResponse)
async def process_video(filename: str):
    model_name = 'yolox_x'
    checkpoints_path = join('/', 'home', 'mvp280102', 'models', 'yolox')
    input_size = (480, 480)
    line_data = (10, (0, 450))
    frames_skip = 0
    filter_label = 0

    input_path = join(inputs.directory, filename)

    watcher = EventWatcher(frames_skip, filter_label)
    await watcher.watch_events(input_path)

    # processor = FrameProcessor(model_name, checkpoints_path, input_size, line_data, frames_skip, filter_label)
    # await processor.process_video(input_path)

    # return RedirectResponse(url=f'/save?filename={filename}')


@app.post('/save')
async def save_events(filename: str):
    database_url = f'{sql_dialect}+{db_driver}://{db_user}:{db_user_password}@{host_name}/{db_name}'
    queue_name = 'vew_events'

    saver = EventSaver(database_url, queue_name)
    await saver.save_events(filename)
