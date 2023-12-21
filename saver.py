import numpy

from sqlalchemy import create_engine, Column, ARRAY, INTEGER, VARCHAR, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Session

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
    def __init__(self, database_url, path, events):
        self.engine = create_engine(database_url)
        BaseModel.metadata.create_all(bind=self.engine)
        register_adapter(numpy.int64, lambda int64: AsIs(int64))

        self.path = path
        self.raw_events = events

    def save_events(self):
        with Session(autoflush=False, bind=self.engine) as session:
            for raw_event in self.raw_events:
                raw_event['filename'] = self.path
                event = Event(**raw_event)
                session.add(event)
                session.commit()
