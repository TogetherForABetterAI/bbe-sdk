"""
BlackBoxSession: Main entry point for the BBE SDK.

This module orchestrates the three main components:
- Connect: Authentication, user information, and middleware setup
- IncomingData: Processing incoming data batches
- OutcomingData: Sending predictions
"""

import logging
from src.utils.logger import initialize_logging
from src.sdk.connect import Connect, InvalidTokenError
from src.sdk.incoming_data import IncomingData
from src.sdk.outcoming_data import OutcomingData


class BlackBoxSession:
    """
    Main session class that orchestrates the SDK components.

    This class integrates Connect, IncomingData, and OutcomingData components
    to provide a seamless experience for the user.
    """

    def __init__(self, eval_input_batch, token, client_id):
        """
        Initialize the BlackBox session.

        Args:
            eval_input_batch: User-provided callback function for model inference
            token: Authentication token
            client_id: Client identifier
        """
        initialize_logging("INFO")
        try:
            # Connect - Handle authentication, user info, and middleware setup
            self._connect = Connect(token=token, client_id=client_id)

            # Store user information
            self._client_id = self._connect.client_id
            self._username = self._connect.username
            self._email = self._connect.email
            self._model_type = self._connect.model_type
            self._inputs_format = self._connect.inputs_format
            self._outputs_format = self._connect.outputs_format

            # Get middleware from Connect component
            self._middleware = self._connect.middleware

            # IncomingData - Handle data processing
            self._incoming_data = IncomingData(
                inputs_format=self._inputs_format,
                on_message_callback=eval_input_batch,
                client_id=client_id,
            )

            # Setup consumption from the client's incoming data queue
            self._middleware.basic_consume(
                self._incoming_data.income_queue, self._handle_incoming_data
            )
            logging.info(
                f"action: setup_consumption | result: success | queue: {self._incoming_data.income_queue}"
            )

            # OutcomingData - Handle sending predictions
            self._outcoming_data = OutcomingData(
                middleware=self._middleware, client_id=client_id
            )
            logging.info(
                f"action: setup_outcoming | result: success | queue: {self._outcoming_data.outcome_queue}"
            )

            # Start consuming immediately upon initialization
            self._middleware.start_consuming()

        except Exception as e:
            logging.error(f"Failed to initialize BlackBoxSession: {e}")
            raise

    def _handle_incoming_data(self, ch, method, properties, body):
        """
        Callback function for middleware when data is received.

        This method coordinates between IncomingData and OutcomingData components.

        Args:
            ch: Channel (from middleware)
            method: Method (from middleware)
            properties: Properties (from middleware)
            body: Raw message body
        """
        try:
            # Process incoming data and get predictions with session_id
            predictions, is_last_batch, batch_index, session_id = (
                self._incoming_data.process_data_batch(body)
            )

            # Send predictions back with session_id
            self._outcoming_data.send_predictions(
                predictions=predictions,
                is_last_batch=is_last_batch,
                batch_index=batch_index,
                session_id=session_id,
            )
        except Exception as e:
            logging.error(f"action: handle_data | result: fail | error: {e}")
            self._middleware.close()
            raise
