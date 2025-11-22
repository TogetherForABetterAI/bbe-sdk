"""
IncomingData component: Handles receiving and processing incoming data batches.
"""

import logging
import numpy as np
from typing import Callable, Any
from src.pb.incomingData.data_batch_pb2 import DataBatchUnlabeled


class IncomingData:
    """
    Manages incoming data processing.

    Responsibilities:
    - Receive data batches
    - Parse and validate data format
    - Transform data (reshape, transpose)
    - Invoke user callback for inference
    """

    def __init__(self, inputs_format, on_message_callback: Callable, client_id: str):
        """
        Initialize the incoming data handler.

        Args:
            inputs_format: Expected format of input data (with dtype and shape)
            on_message_callback: User-provided callback function for model inference
            client_id: Client identifier for queue naming
        """
        self.inputs_format = inputs_format
        self.on_message_callback = on_message_callback
        self.client_id = client_id
        self.income_queue = f"{client_id}_dispatcher_queue"
        self._processed_batch_indices = (
            set()
        )  # Track processed batch indices to avoid duplicates

    def process_data_batch(self, body: bytes) -> tuple[np.ndarray, bool, int, str]:
        """
        Process an incoming data batch.

        Args:
            body: Raw protobuf message body

        Returns:
            Tuple of (predictions, is_last_batch, batch_index, session_id)
            Returns None if batch was already processed (duplicate)

        Raises:
            ValueError: If data format is invalid or processing fails
        """
        # Parse protobuf message
        data_batch = DataBatchUnlabeled()
        data_batch.ParseFromString(body)

        # Check if this batch was already processed
        if data_batch.batch_index in self._processed_batch_indices:
            logging.warning(
                f"action: receive_data_batch | result: duplicate | batch_index: {data_batch.batch_index} | "
                f"session_id: {data_batch.session_id} | Skipping duplicate batch"
            )
            return None  # Return None to indicate duplicate batch

        logging.info(
            f"action: receive_data_batch | result: success | size: {len(body)} | "
            f"eof: {data_batch.is_last_batch} | session_id: {data_batch.session_id} | "
            f"batch_index: {data_batch.batch_index}"
        )

        if not self.inputs_format:
            raise ValueError(
                "Input format not properly configured. Cannot process data without format specification."
            )

        # Process the data
        data_array = self._parse_data(data_batch.data)
        data_array = self._reshape_data(data_array)
        data_array = self._transpose_if_needed(data_array)

        # Invoke user callback for inference
        try:
            predictions = self.on_message_callback(data_array)
        except Exception as e:
            logging.error(f"action: model_inference | result: fail | error: {e}")
            raise

        # Mark this batch as processed
        self._processed_batch_indices.add(data_batch.batch_index)

        return (
            predictions,
            data_batch.is_last_batch,
            data_batch.batch_index,
            data_batch.session_id,
        )

    def _parse_data(self, data: bytes) -> np.ndarray:
        """
        Parse raw data bytes into numpy array.

        Args:
            data: Raw data bytes

        Returns:
            Parsed numpy array

        Raises:
            ValueError: If parsing fails
        """
        logging.debug(
            f"action: process_data_batch | result: in_progress | dtype: {self.inputs_format.dtype} | shape: {self.inputs_format.shape}"
        )

        try:
            data_array = np.frombuffer(data, dtype=self.inputs_format.dtype)
        except Exception as e:
            logging.error(f"action: parse_data_buffer | result: fail | error: {e}")
            raise ValueError(
                f"Failed to parse data buffer with dtype {self.inputs_format.dtype}: {e}"
            )

        return data_array

    def _reshape_data(self, data_array: np.ndarray) -> np.ndarray:
        """
        Reshape data array according to expected format.

        Args:
            data_array: Flat numpy array

        Returns:
            Reshaped numpy array

        Raises:
            ValueError: If reshaping fails or data size is incompatible
        """
        data_size = np.prod(self.inputs_format.shape)
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
            data_array = data_array.reshape((num_samples, *self.inputs_format.shape))
            logging.debug(
                f"action: reshape_data | result: success | final_shape: {data_array.shape}"
            )
        except Exception as e:
            logging.error(f"action: reshape_data | result: fail | error: {e}")
            raise ValueError(f"Failed to reshape data to expected format: {e}")

        return data_array

    def _transpose_if_needed(self, data_array: np.ndarray) -> np.ndarray:
        """
        Transpose data from HWC to CHW format for PyTorch models if needed.

        Args:
            data_array: Numpy array in (batch, H, W, C) format

        Returns:
            Transposed numpy array in (batch, C, H, W) format if applicable

        Raises:
            ValueError: If transposition fails
        """
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

        return data_array
