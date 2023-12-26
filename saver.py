import json
import numpy

from sqlalchemy import create_engine, Column, ARRAY, INTEGER, VARCHAR, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Session

from pika import BlockingConnection, ConnectionParameters
from psycopg2.extensions import register_adapter, AsIs


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


# TODO: Logger.
class EventSaver:
    def __init__(self, database_url, queue_name, host_name='localhost'):
        self.engine = create_engine(database_url)
        BaseModel.metadata.create_all(bind=self.engine)
        register_adapter(numpy.int64, lambda int64: AsIs(int64))

        self.queue_name = queue_name
        self.host_name = host_name

    def save_events(self, filename):
        connection = BlockingConnection(ConnectionParameters(host=self.host_name))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)

        while True:
            method_frame, _, body = channel.basic_get(queue=self.queue_name)

            if not method_frame:
                break

            with Session(autoflush=False, bind=self.engine) as session:
                raw_event = json.loads(body)
                raw_event['filename'] = filename
                event = Event(**raw_event)
                session.add(event)
                session.commit()

            channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        connection.close()
