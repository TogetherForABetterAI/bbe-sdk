import pika
import logging


class Middleware:
    def __init__(self, host, port, routing_key):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port)
        )
        self._connection = connection
        channel = connection.channel()
        channel.confirm_delivery()
        channel.basic_qos(prefetch_count=1)
        self._channel = channel
        self._routing_key = routing_key

    def basic_send(self, message: str = "", exchange_name: str = ""):
        try:
            self._channel.basic_publish(
                exchange=exchange_name,
                routing_key=self._routing_key,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent
                ),
            )

        except Exception as e:
            logging.error(
                f"action: message_sending | result: fail | error: {e} | routing_key: {self._routing_key}"
            )

    def basic_consume(self, queue_name: str, callback_function):
        self._channel.basic_consume(
            queue=queue_name,
            on_message_callback=self.callback_wrapper(callback_function),
        )

    def callback_wrapper(self, callback_function):
        def wrapper(ch, method, properties, body):
            callback_function(ch, method, properties, body)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        return wrapper

    def start(self):
        try:
            self._channel.start_consuming()
        except OSError:
            self.close()

    def close(self):
        try:
            self._channel.close()
            self._connection.close()
        except Exception as e:
            logging.error(
                f"action: rabbitmq_connection_close | result: fail | error: {e}"
            )
