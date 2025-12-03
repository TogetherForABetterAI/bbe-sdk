"""
OutcomingData component: Handles sending predictions back to the service.
"""

import logging
from typing import Optional, Union, List
import numpy as np
from src.config.config import CALIBRATON_QUEUE
from src.pb.outcomingData import sdk_outcoming_data_pb2


class OutcomingData:
    """
    Manages outgoing predictions.

    Responsibilities:
    - Format predictions into protobuf messages
    - Send predictions through middleware
    """

    def __init__(self, middleware, user_id: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the outcoming data handler.

        Args:
            middleware: Middleware instance for sending messages
            user_id: User identifier for queue naming
        """
        self.middleware = middleware
        self.user_id = user_id
        self.outcome_queue = CALIBRATON_QUEUE % user_id
        base_logger = logger or logging.getLogger(__name__)
        self.logger = logging.LoggerAdapter(
            base_logger, 
            {'user_id': user_id, 'component': 'OutcomingData'}
        )

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
        
        pred_shape = "N/A"
        if isinstance(predictions, np.ndarray):
            pred_shape = str(predictions.shape)
        elif isinstance(predictions, list):
            pred_shape = f"List[len={len(predictions)}]"

        self.logger.debug("Formatting predictions", extra={
            'batch_index': batch_index,
            'data_type': type(predictions).__name__,
            'shape': pred_shape
        })
        try:
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
        except Exception as e:

            self.logger.error("Failed to serialize predictions. Check output format.", extra={
                'hint': 'Ensure callback returns a list of lists or 2D numpy array',
                'error': str(e)
            })
            raise 
        
        self.logger.info("Sending predictions", extra={
            'batch_index': batch_index,
            'payload_size_bytes': len(batch),
            'is_eof': is_last_batch,
            'queue': self.outcome_queue
        })
        try:
            # Send directly to the client's calibration queue
            self.middleware.publish(message=batch, queue_name=self.outcome_queue)
        except Exception as e:
            self.logger.error("Failed to publish message", exc_info=True, extra={
                'queue': self.outcome_queue
            })
            raise

        if is_last_batch:
            self.logger.info("Session complete. Stopping consumption.", extra={
                'session_id': session_id,
                'action': 'stop_consuming'
            })
            self.middleware.stop_consuming()