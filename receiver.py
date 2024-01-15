import json
import asyncio

from pika import BlockingConnection, ConnectionParameters

from utils import create_logger


class EventReceiver:
    logger = create_logger(__name__)

    def __init__(self, config, extractor, saver):
        self.event_names = config.event_names

        self.host_name = config.host_name
        self.queue_name = config.queue_name
        self.timeout = config.timeout

        self.extractor = extractor
        self.saver = saver

    # actions - iterable of async callable:
    async def receive_events(self):
        connection = BlockingConnection(ConnectionParameters(host=self.host_name))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)

        while True:
            await asyncio.sleep(self.timeout)

            event, tag = self._receive_event(channel)

            self.logger.info("Receive event message {} from '{}' queue.".format(event, self.queue_name))

            if not tag:
                break

            if event['event_name'] not in self.event_names:
                self.logger.info("Skip inappropriate event message with name '{}'.".format(event['event_name']))
            else:
                await self.extractor.extract_event(event)
                await self.saver.save_event(event)

            channel.basic_ack(delivery_tag=tag)

        connection.close()

    def _receive_event(self, channel):
        method_frame, _, body = channel.basic_get(queue=self.queue_name)

        if not method_frame:
            return None, None

        tag = method_frame.delivery_tag
        event = json.loads(body)

        return event, tag
