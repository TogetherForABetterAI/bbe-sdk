"""
OutcomingData component: Handles sending predictions back to the service.
"""

import logging
from typing import Union, List
import numpy as np
from src.pb.outcomingData import calibration_pb2
from src.utils.config import REPLIES_EXCHANGE


class OutcomingData:
    """
    Manages outgoing predictions.

    Responsibilities:
    - Format predictions into protobuf messages
    - Send predictions through middleware
    """

    def __init__(self, middleware):
        """
        Initialize the outcoming data handler.

        Args:
            middleware: Middleware instance for sending messages
        """
        self.middleware = middleware

    def send_predictions(
        self,
        predictions: Union[List[float], np.ndarray],
        is_last_batch: bool,
        batch_index: int,
    ) -> None:
        """
        Send predictions back to the service.

        Args:
            predictions: Model predictions (probabilities)
            is_last_batch: Flag indicating if this is the last batch
            batch_index: Index of the current batch

        Raises:
            Exception: If sending fails
        """
        # Create protobuf message
        pred = calibration_pb2.Predictions()

        for prob in predictions:
            pred_list = calibration_pb2.PredictionList()
            pred_list.values.extend(prob)
            pred.pred.append(pred_list)

        pred.eof = is_last_batch
        pred.batch_index = batch_index

        # Serialize and send
        batch = pred.SerializeToString()

        logging.info(
            f"action: send_probs | result: success | size: {len(batch)} | eof: {pred.eof}"
        )

        self.middleware.basic_send(message=batch, exchange_name=REPLIES_EXCHANGE)
