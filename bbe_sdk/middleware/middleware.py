import pika
import logging
from bbe_sdk.utils.config import DATASET_EXCHANGE, REPLIES_EXCHANGE, config_params


class Middleware:
    def __init__(self, host, port, routing_key):
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, port=port)
            )
        except Exception as e:
            logging.error(
                f"action: rabbitmq_connection_init | result: fail | error: {e}"
            )
            raise
        self._connection = connection
        channel = connection.channel()
        channel.confirm_delivery()
        channel.basic_qos(prefetch_count=1)
        self._channel = channel
        self._routing_key = routing_key

    def declare_queue(self, queue_name: str, durable: bool = False):
        self._channel.queue_declare(queue=queue_name, durable=durable)

    def bind_queue(self, queue_name, exchange_name, routing_key):
        self._channel.queue_bind(
            exchange=exchange_name, queue=queue_name, routing_key=routing_key
        )

    def declare_exchange(
        self, exchange_name: str, exchange_type: str = "direct", durable: bool = False
    ):
        try:
            self._channel.exchange_declare(
                exchange=exchange_name, exchange_type=exchange_type, durable=durable
            )
            logging.info(f"Exchange '{exchange_name}' declared successfully")
        except Exception as e:
            logging.error(f"Failed to declare exchange '{exchange_name}': {e}")
            raise e

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

    @staticmethod
    def setup_client_middleware(client_id, callback_function):
        try:
            middleware = Middleware(
                host=config_params["rabbitmq_host"],
                port=config_params["rabbitmq_port"],
                routing_key=client_id,
            )

            middleware.declare_exchange(DATASET_EXCHANGE, "direct", False)
            middleware.declare_exchange(REPLIES_EXCHANGE, "direct", False)
            middleware.declare_queue(f"{client_id}_unlabeled_queue", False)
            middleware.bind_queue(
                f"{client_id}_unlabeled_queue",
                DATASET_EXCHANGE,
                f"{client_id}.unlabeled",
            )
            middleware.basic_consume(f"{client_id}_unlabeled_queue", callback_function)

            return middleware
        except Exception as e:
            logging.error(
                f"action: client_middleware_setup | result: fail | error: {e}"
            )
