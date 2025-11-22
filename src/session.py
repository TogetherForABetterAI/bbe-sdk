"""
BlackBoxSession: Main entry point for the BBE SDK.

This module orchestrates the three main components:
- Connect: Authentication and user information
- IncomingData: Processing incoming data batches
- OutcomingData: Sending predictions
"""

import logging
from src.middleware.middleware import Middleware
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

        # Connect - Handle authentication and user info
        self._connect = Connect(token=token, client_id=client_id)

        # Store user information
        self._client_id = self._connect.client_id
        self._username = self._connect.username
        self._email = self._connect.email
        self._model_type = self._connect.model_type
        self._inputs_format = self._connect.inputs_format
        self._outputs_format = self._connect.outputs_format

        # IncomingData - Handle data processing
        self._incoming_data = IncomingData(
            inputs_format=self._inputs_format, on_message_callback=eval_input_batch
        )

        # Setup middleware with callback to our data handler
        try:
            self._middleware = Middleware.setup_client_middleware(
                client_id=client_id, callback_function=self._handle_incoming_data
            )
        except Exception as e:
            logging.error(f"Failed to set up middleware: {e}")
            raise

        # OutcomingData - Handle sending predictions
        self._outcoming_data = OutcomingData(middleware=self._middleware)

        # Start consuming immediately upon initialization
        try:
            self._middleware.start()
        except Exception as e:
            logging.error(f"Failed to start BlackBoxSession: {e}")
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
            # Process incoming data and get predictions
            predictions, is_last_batch, batch_index = (
                self._incoming_data.process_data_batch(body)
            )

            # Send predictions back
            self._outcoming_data.send_predictions(
                predictions=predictions,
                is_last_batch=is_last_batch,
                batch_index=batch_index,
            )
        except Exception as e:
            logging.error(f"action: handle_data | result: fail | error: {e}")
            self._middleware.close()
            raise
