import pika
import logging


class Middleware:
    def __init__(self, host, port, username, password):
        """
        Initialize middleware with RabbitMQ credentials.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
        """
        try:
            credentials = pika.PlainCredentials(username, password)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, port=port, credentials=credentials)
            )
        except Exception as e:
            logging.error(
                f"action: rabbitmq_connection_init | result: fail | error: {e}"
            )
            raise
        self._connection = connection
        self._channel = connection.channel()
        self._channel.confirm_delivery()
        self._channel.basic_qos(prefetch_count=1)

    def publish(self, message: str, queue_name: str):
        """
        Send a message to RabbitMQ.

        Args:
            message: Message body to send
            queue_name: Queue name (for direct queue publishing)

        Note:
            If queue_name is provided, sends directly to the queue (exchange="" and routing_key=queue_name).
        """
        try:
            # Send directly to queue (default exchange with queue name as routing key)
            self._channel.basic_publish(
                exchange="",  # Default exchange
                routing_key=queue_name,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent
                ),
            )
        except Exception as e:
            logging.error(
                f"action: message_sending | result: fail | error: {e} | "
                f"queue: {queue_name} | size: {len(message)}"
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

    def start_consuming(self):

        if self._channel and not self._channel.is_closed:
            self._channel.start_consuming()

    def close(self):
        try:
            if self._channel and not self._channel.is_closed:
                self._channel.close()

            if self._connection:
                self._connection.close()
        except Exception as e:
            logging.error(
                f"action: rabbitmq_connection_close | result: fail | error: {e}"
            )
