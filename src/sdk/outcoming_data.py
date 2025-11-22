"""
OutcomingData component: Handles sending predictions back to the service.
"""

import logging
from typing import Union, List
import numpy as np
from src.pb.outcomingData import sdk_outcoming_data_pb2


class OutcomingData:
    """
    Manages outgoing predictions.

    Responsibilities:
    - Format predictions into protobuf messages
    - Send predictions through middleware
    """

    def __init__(self, middleware, client_id: str):
        """
        Initialize the outcoming data handler.

        Args:
            middleware: Middleware instance for sending messages
            client_id: Client identifier for queue naming
        """
        self.middleware = middleware
        self.client_id = client_id
        self.outcome_queue = f"{client_id}_calibration_queue"

    def send_predictions(
        self,
        predictions: Union[List[float], np.ndarray],
        is_last_batch: bool,
        batch_index: int,
        session_id: str,
    ) -> None:
        """
        Send predictions back to the service.

        Args:
            predictions: Model predictions (probabilities)
            is_last_batch: Flag indicating if this is the last batch
            batch_index: Index of the current batch
            session_id: Session identifier

        Raises:
            Exception: If sending fails
        """
        # Create protobuf message
        pred = sdk_outcoming_data_pb2.Predictions()

        for prob in predictions:
            pred_list = sdk_outcoming_data_pb2.PredictionList()
            pred_list.values.extend(prob)
            pred.pred.append(pred_list)

        pred.eof = is_last_batch
        pred.batch_index = batch_index
        pred.session_id = session_id

        # Serialize and send
        batch = pred.SerializeToString()

        logging.info(
            f"action: send_probs | result: success | size: {len(batch)} | "
            f"eof: {pred.eof} | session_id: {session_id} | queue: {self.outcome_queue}"
        )

        # Send directly to the client's calibration queue
        self.middleware.publish(message=batch, queue_name=self.outcome_queue)

        # Stop consuming if this is the last batch
        if is_last_batch:
            logging.info(
                f"Last batch sent. Stopping consumption for session: {session_id}"
            )
            self.middleware.stop_consuming()
