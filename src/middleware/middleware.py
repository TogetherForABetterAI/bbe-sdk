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

    def basic_consume(self, queue_name: str, callback_function, consumer_tag: str = None):
        """
        Start consuming messages from a queue.

        Args:
            queue_name: Name of the queue to consume from
            callback_function: Callback function to handle messages
        """
        self._channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback_function,
            auto_ack=False,  # Manual acknowledgment
            consumer_tag=consumer_tag
        )

    def ack_message(self, channel: pika.channel.Channel, delivery_tag: int):
        """
        Acknowledge a message.

        Args:
            channel: The channel to use
            delivery_tag: The delivery tag of the message to ACK
        """
        try:
            if channel and not channel.is_closed:
                channel.basic_ack(delivery_tag=delivery_tag)
                logging.debug(f"ACKed message with delivery_tag={delivery_tag}")
            else:
                logging.warning(f"Cannot ACK message: channel is closed")
        except Exception as e:
            logging.error(f"Failed to ACK message {delivery_tag}: {e}")
            raise

    def nack_message(
        self, channel: pika.channel.Channel, delivery_tag: int, requeue: bool = False
    ):
        """
        Negative acknowledge a message (NACK).

        Args:
            channel: The channel to use
            delivery_tag: The delivery tag of the message to NACK
            requeue: Whether to requeue the message (True) or discard it (False)
        """
        try:
            if channel and not channel.is_closed:
                channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
                logging.debug(
                    f"NACKed message with delivery_tag={delivery_tag}, requeue={requeue}"
                )
            else:
                logging.warning(f"Cannot NACK message: channel is closed")
        except Exception as e:
            logging.error(
                f"Failed to NACK message {delivery_tag} (requeue={requeue}): {e}"
            )
            raise

    def stop_consuming(self):
        """Stop consuming messages."""
        try:
            if self._channel and not self._channel.is_closed:
                self._channel.stop_consuming()
                logging.info("Stopped consuming messages")
        except Exception as e:
            logging.error(f"Failed to stop consuming: {e}")
            raise

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
