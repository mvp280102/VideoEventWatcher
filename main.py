# TODO: Remove temporary file.


from os.path import join
from aiofiles import open

from fastapi import FastAPI, File, UploadFile, HTTPException, status

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from processor import VideoProcessor
from cacher import EventCacher
from saver import EventSaver


MAX_FILE_SIZE = 10 * 1024 * 1024


inputs = StaticFiles(directory='inputs')
outputs = StaticFiles(directory='outputs')

app = FastAPI()
app.mount('/inputs', inputs, name='inputs')
app.mount('/outputs', outputs, name='outputs')


@app.post('/upload', response_class=RedirectResponse)
async def create_file(file_object: UploadFile = File(...)):
    file_path = join(inputs.directory, file_object.filename)

    async with open(file_path, 'wb') as buffer:
        content = await file_object.read()

        if len(content) >= MAX_FILE_SIZE:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f'File is too large. Size limit is {int(MAX_FILE_SIZE / (1024 * 1024))} MB.')

        await buffer.write(content)
        await buffer.close()

    return RedirectResponse(url=f'/process?filename={file_object.filename}')


@app.post('/process')
async def process_video(filename: str):
    model_name = 'yolox_x'
    checkpoints_path = join('/', 'home', 'mvp280102', 'models', 'yolox')
    input_size = (480, 480)
    frames_skip = 0
    filter_label = 0

    input_path = join(inputs.directory, filename)

    processor = VideoProcessor(model_name, checkpoints_path, input_size, frames_skip, filter_label)
    processor.process_video(input_path)

    database_url = 'postgresql+psycopg2://videologger:videologger@localhost/videologger'
    events = processor.events

    logger = EventSaver(database_url, filename, events)
    logger.log_events()

    return None
