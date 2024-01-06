import json
import asyncio
import numpy as np

from sqlalchemy import create_engine, Column, INTEGER, VARCHAR, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Session

from pika import BlockingConnection, ConnectionParameters
from psycopg2.extensions import register_adapter, AsIs

from utils import create_logger


class BaseModel(DeclarativeBase):
    pass


class Event(BaseModel):
    __tablename__ = 'events'

    event_id = Column(INTEGER, nullable=False, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=False), nullable=False)
    video_path = Column(VARCHAR(length=128), nullable=False)
    tracks_path = Column(VARCHAR(length=128), nullable=False)
    frame_index = Column(INTEGER, nullable=False)
    track_id = Column(INTEGER, nullable=False)
    event_name = Column(VARCHAR(length=128), nullable=False)


class EventSaver:
    logger = create_logger(__name__, stream=False)

    def __init__(self, config):
        database_url = ('{sql_dialect}+{db_driver}'
                        '://{db_user}:{db_user_password}'
                        '@{host_name}/{db_name}').format(**config.credentials)

        self.engine = create_engine(database_url)
        BaseModel.metadata.create_all(bind=self.engine)
        register_adapter(np.int64, lambda int64: AsIs(int64))

        self.timeout = config.timeout
        self.queue_name = config.queue_name
        self.host_name = config.credentials.host_name

    async def save_events(self):
        connection = BlockingConnection(ConnectionParameters(host=self.host_name))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)

        while True:
            await asyncio.sleep(self.timeout)

            method_frame, _, body = channel.basic_get(queue=self.queue_name)

            if not method_frame:
                break

            raw_event = json.loads(body)

            self.logger.info("Receive event message {} from '{}' queue.".format(raw_event, self.queue_name))

            with Session(autoflush=False, bind=self.engine) as session:
                event = Event(**raw_event)
                session.add(event)
                session.commit()
                # TODO: Split messages receiving and saving to DB.
                self.logger.info("Write event object {} to database.".format(event))

            channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        connection.close()
