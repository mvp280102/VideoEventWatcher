import json

from pika import BlockingConnection, ConnectionParameters, BasicProperties, DeliveryMode

from utils import create_logger


class EventSender:
    logger = create_logger(__name__)

    def __init__(self, queue_name, host_name='localhost'):
        self.queue_name = queue_name
        self.host_name = host_name

    def send_events(self, events):
        if not events:
            return

        connection = BlockingConnection(ConnectionParameters(self.host_name))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name, durable=True)

        for event in events:
            self.logger.info(f'Send event message {event} to message queue.')
            channel.basic_publish(exchange='', routing_key=self.queue_name, body=json.dumps(event),
                                  properties=BasicProperties(delivery_mode=DeliveryMode.Persistent))

        connection.close()
