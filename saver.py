import json
import numpy
import asyncio

from sqlalchemy import create_engine, Column, ARRAY, INTEGER, VARCHAR, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Session

from pika import BlockingConnection, ConnectionParameters
from psycopg2.extensions import register_adapter, AsIs

from utils import create_logger


class BaseModel(DeclarativeBase):
    pass


class Event(BaseModel):
    __tablename__ = 'logs'

    event_id = Column(INTEGER, nullable=False, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=False), nullable=False)
    event_name = Column(VARCHAR(length=64), nullable=False)
    track_id = Column(INTEGER, nullable=False)
    position = Column(ARRAY(item_type=INTEGER, as_tuple=True), nullable=False)
    filename = Column(VARCHAR(length=64), nullable=False)
    frame_path = Column(VARCHAR(length=256), nullable=True)


class EventSaver:
    logger = create_logger(__name__)

    def __init__(self, database_url, queue_name, host_name='localhost', timeout=1.0):
        self.engine = create_engine(database_url)
        BaseModel.metadata.create_all(bind=self.engine)
        register_adapter(numpy.int64, lambda int64: AsIs(int64))

        self.queue_name = queue_name
        self.host_name = host_name
        self.timeout = timeout

    async def save_events(self, filename):
        connection = BlockingConnection(ConnectionParameters(host=self.host_name))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)

        while True:
            await asyncio.sleep(self.timeout)

            method_frame, _, body = channel.basic_get(queue=self.queue_name)

            if not method_frame:
                break

            raw_event = json.loads(body)

            self.logger.info(f'Receive event message {raw_event} from message queue.')

            raw_event['filename'] = filename

            with Session(autoflush=False, bind=self.engine) as session:
                event = Event(**raw_event)
                session.add(event)
                session.commit()
                self.logger.info(f'Write event object {event} to database.')

            channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        connection.close()
