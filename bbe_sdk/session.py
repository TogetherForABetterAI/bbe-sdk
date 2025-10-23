from typing import Union, List
import numpy as np
from bbe_sdk.middleware.middleware import Middleware
import logging
from bbe_sdk.proto import calibration_pb2
from bbe_sdk.pb.dataset_service.dataset_service_pb2 import DataBatchUnlabeled
from bbe_sdk.utils.logger import initialize_logging
from bbe_sdk.utils.config import REPLIES_EXCHANGE, config_params
from bbe_sdk.utils.data import parse_inputs_format
import requests


class InvalidTokenError(Exception):
    pass


class BlackBoxSession:
    # Internal vars:
    # - client_id
    # - username
    # - email
    # - mode_type
    # - inputs_format
    def __init__(self, eval_input_batch, token, client_id):
        initialize_logging("INFO")
        try:
            validate_token_resp = requests.post(
                "http://users-service:8000/tokens/validate",
                json={"token": token, "client_id": client_id},
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to users server: {e}")
        if validate_token_resp.status_code != 200:
            raise InvalidTokenError(
                f"Users server returned status {validate_token_resp.status_code}: {validate_token_resp.text}"
            )

        validate_token_data = validate_token_resp.json()
        if not validate_token_data.get("is_valid", False):
            raise InvalidTokenError("Token validation failed: not authorized.")

        try:
            user_info_resp = requests.get(
                f"http://users-service:8000/users/{client_id}",
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to users server: {e}")
        if user_info_resp.status_code != 200:
            raise InvalidTokenError(
                f"Users server returned status {user_info_resp.status_code}: {user_info_resp.text}"
            )

        user_info_data = user_info_resp.json()  # of type GetUserDataResponse
        self._client_id = user_info_data.get("client_id", "")
        self._username = user_info_data.get("username", "")
        self._email = user_info_data.get("email", "")
        self._model_type = user_info_data.get("model_type", "")
        inputs_format_str = user_info_data.get("inputs_format", "")
        self._inputs_format = parse_inputs_format(inputs_format_str)

        if not self._inputs_format:
            raise RuntimeError(
                f"System configuration error: Invalid input format specification '{inputs_format_str}' for user {client_id}."
            )

        self._outputs_format = user_info_data.get("outputs_format", "")

        logging.debug(
            f"action: receive_user_info | result: success | User info: {self._client_id}, {self._username}, {self._email}, {self._model_type}, {self._inputs_format}, {self._outputs_format}"
        )

        # Connect to the users-service
        try:
            connect_resp = requests.post(
                "http://users-service:8000/users/connect",
                json={"client_id": client_id, "token": token},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to users-service: {e}")
        if connect_resp.status_code != 200:
            raise InvalidTokenError(
                f"Connection service returned status {connect_resp.status_code}: {connect_resp.text}"
            )

        connect_data = connect_resp.json()
        if connect_data.get("status") != "success":
            raise InvalidTokenError(
                f"Connection failed: {connect_data.get('message', 'Unknown error')}"
            )

        logging.info(
            f"action: connect_to_service | result: success | client_id: {client_id}"
        )

        self._on_message = eval_input_batch
        try:
            self._middleware = Middleware.setup_client_middleware(
                client_id=client_id, callback_function=self.get_data
            )
        except Exception as e:
            logging.error(f"Failed to set up middleware: {e}")
            raise
        self._count = 0

        # Start consuming immediately upon initialization
        try:
            self._middleware.start()
        except Exception as e:
            logging.error(f"Failed to start BlackBoxSession: {e}")
            raise

    def get_data(self, ch, method, properties, body):
        self._count += 1
        data_batch = DataBatchUnlabeled()
        data_batch.ParseFromString(body)
        logging.info(
            f"action: receive_data_batch | result: success | size: {len(body)} | eof: {data_batch.is_last_batch}"
        )

        if not self._inputs_format:
            raise ValueError(
                "Input format not properly configured. Cannot process data without format specification."
            )

        data_size = np.prod(self._inputs_format.shape)

        logging.debug(
            f"action: process_data_batch | result: in_progress | dtype: {self._inputs_format.dtype} | shape: {self._inputs_format.shape}"
        )

        try:
            data_array = np.frombuffer(data_batch.data, dtype=self._inputs_format.dtype)
        except Exception as e:
            logging.error(f"action: parse_data_buffer | result: fail | error: {e}")
            raise ValueError(
                f"Failed to parse data buffer with dtype {self._inputs_format.dtype}: {e}"
            )

        num_elements = data_array.size
        num_samples = num_elements // data_size

        if num_samples * data_size != num_elements:
            raise ValueError(
                f"Data size incompatible with expected format. "
                f"Expected elements per sample: {data_size}, "
                f"total elements: {num_elements}, "
                f"calculated samples: {num_samples}, "
                f"remainder: {num_elements % data_size}"
            )

        try:
            data_array = data_array.reshape((num_samples, *self._inputs_format.shape))
            logging.debug(
                f"action: reshape_data | result: success | final_shape: {data_array.shape}"
            )
        except Exception as e:
            logging.error(f"action: reshape_data | result: fail | error: {e}")
            raise ValueError(f"Failed to reshape data to expected format: {e}")

        # Convert from HWC to CHW format for PyTorch models
        # If shape is (batch, H, W, C) and C is 1 or 3, transpose to (batch, C, H, W)
        if len(data_array.shape) == 4 and data_array.shape[-1] in [1, 3]:
            try:
                # Transpose from (batch, H, W, C) to (batch, C, H, W)
                data_array = np.transpose(data_array, (0, 3, 1, 2))
                logging.debug(
                    f"action: transpose_to_chw | result: success | final_shape: {data_array.shape}"
                )
            except Exception as e:
                logging.error(f"action: transpose_to_chw | result: fail | error: {e}")
                raise ValueError(f"Failed to transpose data to CHW format: {e}")

        try:
            probs = self._on_message(data_array)
        except Exception as e:
            logging.error(f"action: model_inference | result: fail | error: {e}")
            raise

        try:
            self._send_probs(probs, data_batch.is_last_batch, data_batch.batch_index)
        except Exception as e:
            logging.error(f"action: send_probs | result: fail | error: {e}")
            self._middleware.close()
            raise

    def _send_probs(
        self,
        probs: Union[List[float], np.ndarray],
        is_last_batch: bool,
        batch_index: int,
    ) -> None:
        pred = calibration_pb2.Predictions()
        for prob in probs:
            pred_list = calibration_pb2.PredictionList()
            pred_list.values.extend(prob)
            pred.pred.append(pred_list)

        pred.eof = is_last_batch
        pred.batch_index = batch_index
        batch = pred.SerializeToString()
        logging.info(
            f"action: send_probs | result: success | size: {len(batch)} | eof: {pred.eof}"
        )
        self._middleware.basic_send(message=batch, exchange_name=REPLIES_EXCHANGE)
