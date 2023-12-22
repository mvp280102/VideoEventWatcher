from pika import BlockingConnection, ConnectionParameters, BasicProperties, DeliveryMode


def send_events(events):
    connection = BlockingConnection(ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='vew_events', durable=True)

    for event in events:
        channel.basic_publish(exchange='', routing_key='vew_events', body=str(event),
                              properties=BasicProperties(delivery_mode=DeliveryMode.Persistent))

    connection.close()
