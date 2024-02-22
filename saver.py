import numpy as np

from sqlalchemy import create_engine, Column, INTEGER, VARCHAR, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Session

from psycopg2.extensions import register_adapter, AsIs

from utils import create_logger


class BaseModel(DeclarativeBase):
    pass


class Event(BaseModel):
    __tablename__ = 'events'

    event_id = Column(INTEGER, nullable=False, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=False), nullable=False)
    file_name = Column(VARCHAR(length=128), nullable=False)
    frame_index = Column(INTEGER, nullable=False)
    track_id = Column(INTEGER, nullable=False)
    event_name = Column(VARCHAR(length=128), nullable=False)


class EventSaver:
    logger = create_logger(__name__)

    def __init__(self, config):
        database_url = ('{sql_dialect}+{db_driver}'
                        '://{db_username}:{db_password}'
                        '@{host_name}/{db_name}').format(**config)

        self.engine = create_engine(database_url)
        BaseModel.metadata.create_all(bind=self.engine)
        register_adapter(np.int64, lambda int64: AsIs(int64))

    def save_events(self, events):
        frame_index = events[0]['frame_index']

        self.logger.newline()
        self.logger.info("Saving event(s) for frame {} to database:".format(frame_index))

        for event in events:
            self._save_event(event)

    def _save_event(self, event):
        with Session(autoflush=False, bind=self.engine) as session:
            event_entry = Event(**event)
            session.add(event_entry)
            session.commit()

            self.logger.info("Track ID - {}, event name - {}.".format(event['track_id'], event['event_name']))
