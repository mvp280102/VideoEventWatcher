import json

from pika import BlockingConnection, ConnectionParameters, BasicProperties, DeliveryMode

from utils import create_logger


class EventSender:
    logger = create_logger(__name__)

    def __init__(self, config):
        self.queue_name = config.queue_name
        self.host_name = config.host_name

    def send_events(self, events):
        if not events:
            return

        connection = BlockingConnection(ConnectionParameters(self.host_name))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)

        for event in events:
            self._send_event(channel, event)

        connection.close()

    def _send_event(self, channel, event):
        channel.basic_publish(exchange='', routing_key=self.queue_name, body=json.dumps(event),
                              properties=BasicProperties(delivery_mode=DeliveryMode.Persistent))

        self.logger.info("Send event message {} to '{}' queue.".format(event, self.queue_name))
