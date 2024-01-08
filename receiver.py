import json
import asyncio

from pika import BlockingConnection, ConnectionParameters

from utils import create_logger


class EventReceiver:
    logger = create_logger(__name__)

    def __init__(self, config):
        self.host_name = config.host_name
        self.queue_name = config.queue_name
        self.timeout = config.timeout

    # actions - iterable of async callable:
    async def receive_events(self, actions=None):
        connection = BlockingConnection(ConnectionParameters(host=self.host_name))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)

        while True:
            await asyncio.sleep(self.timeout)

            event, tag = self.receive_event(channel)

            if not event:
                break

            if actions:
                for action in actions:
                    await action(event)

            channel.basic_ack(delivery_tag=tag)

        connection.close()

    def receive_event(self, channel):
        method_frame, _, body = channel.basic_get(queue=self.queue_name)

        if not method_frame:
            return None, None

        tag = method_frame.delivery_tag
        event = json.loads(body)

        self.logger.info("Receive event message {} from '{}' queue.".format(event, self.queue_name))

        return event, tag
