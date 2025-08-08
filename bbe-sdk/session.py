import asyncio
import base64
from typing import Iterator, Tuple, Union, List

import numpy as np
from .types import Image, Label, Probs
from .middleware.middleware import Middleware
import socket
import json
import logging
from bbe_sdk.proto import calibration_pb2, dataset_pb2
import requests


class InvalidTokenError(Exception):
    pass


class BlackBoxSession:
    def __init__(self, eval_input_batch, token, client_id):
        # Authenticate
        try:
            resp = requests.post(
                "http://authenticator-service-app:8000/tokens/validate",
                json={"token": token, "client_id": client_id},
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to auth server: {e}")
        if resp.status_code != 200:
            raise InvalidTokenError(
                f"Auth server returned status {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        if not data.get("is_valid", False):
            raise InvalidTokenError("Token validation failed: not authorized.")
        connection = data["connection"]
        amqp_host = connection["rabbit_host"]
        amqp_port = connection["rabbit_port"]
        routing_key = connection["routing_key"]
        queue_name = connection["queue_name"]
        self._on_message = eval_input_batch
        self._middleware = Middleware(amqp_host, amqp_port, routing_key)
        self._middleware.basic_consume(queue_name, self.get_data)
        self._count = 0

    def start(self):
        """
        Starts the session and begins consuming messages.
        This method is blocking and will run until the session is closed.
        """
        try:
            self._middleware.start()
        except Exception as e:
            logging.error(f"Failed to start BlackBoxSession: {e}")
            raise

    def get_data(self, ch, method, properties, body):
        self._count += 1
        logging.info(f"body: {body}...")  # Log only the first 100 bytes for brevity
        data_batch = dataset_pb2.DataBatch()
        data_batch.ParseFromString(body)

        image_dtype = np.float32
        image_shape = (1, 28, 28)
        image_size = np.prod(image_shape)

        # Decodificás el buffer de imágenes
        images = np.frombuffer(data_batch.data, dtype=image_dtype)

        num_floats = images.size
        num_images = num_floats // image_size

        if num_images * image_size != num_floats:
            raise ValueError("Tamaño de datos incompatible con imagen")

        images = images.reshape((num_images, *image_shape))

        # Ahora llamás a tu lógica de predicción
        probs = self._on_message(images)

        try:
            self._send_probs(1, probs, data_batch.is_last_batch, data_batch.batch_index)
            logging.info(f"probs_sent: {probs}")
        except Exception as e:
            logging.error(f"action: send_probs | result: fail | error: {e}")
            self._middleware.close()
            raise

        logging.info(f"action: messages_count: {self._count}")

    def _send_probs(
        self,
        label: int,
        probs: Union[List[float], np.ndarray],
        is_last_batch: bool,
        batch_index: int,
    ) -> None:
        pred = calibration_pb2.Predictions()
        for prob in probs:
            pred_list = calibration_pb2.PredictionList()
            pred_list.values.extend(prob)
            pred_list.label = label
            pred.pred.append(pred_list)
        pred.eof = is_last_batch
        pred.batch_index = batch_index
        batch = pred.SerializeToString()
        self._middleware.basic_send(message=batch, exchange_name="calibration-exchange")

    def __iter__(self) -> Iterator[Tuple[Image, int, bool]]:
        return self

    def __next__(self) -> Tuple[Image, int, bool]:
        try:
            return next(self._data_iter)
        except StopIteration:
            if not self._loop.is_closed():
                self._loop.close()
            raise

    def __del__(self):
        if hasattr(self, "_loop") and not self._loop.is_closed():
            self._loop.close()
