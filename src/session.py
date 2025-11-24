"""
BlackBoxSession: Main entry point for the BBE SDK.

This module orchestrates the three main components:
- Connect: Authentication, user information, and middleware setup
- IncomingData: Processing incoming data batches
- OutcomingData: Sending predictions
"""

import logging
from src.config.logger import initialize_logging
from src.sdk.connect import Connect
from src.sdk.incoming_data import IncomingData
from src.sdk.outcoming_data import OutcomingData
from src.models.inputs_format import parse_inputs_format
import numpy as np


class BlackBoxSession:
    """
    Main session class that orchestrates the SDK components.

    This class integrates Connect, IncomingData, and OutcomingData components.
    """

    def __init__(self, eval_input_batch, token, user_id):
        """
        Initialize the BlackBox session.

        Args:
            eval_input_batch: User-provided callback function for model inference
            token: Authentication token
            user_id: Client identifier
        """
        initialize_logging("INFO")
        try:
            # Connect - Handle authentication, user info, and middleware setup
            connect = Connect(token=token, user_id=user_id)

            # Execute complete connection flow and get middleware + connection response
            middleware, inputs_format = connect.try_connect()

            # Parse inputs format
            inputs_format = parse_inputs_format(inputs_format)
            if not inputs_format:
                raise RuntimeError(
                    f"System configuration error: Invalid input format specification '{inputs_format.inputs_format}' for user {user_id}."
                )

            # Store parsed format and middleware
            self._user_id = user_id
            self._inputs_format = inputs_format
            self._middleware = middleware

            # IncomingData - Handle data processing
            self._incoming_data = IncomingData(
                inputs_format=self._inputs_format,
                on_message_callback=eval_input_batch,
                user_id=user_id,
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
                middleware=self._middleware, user_id=user_id
            )
            logging.info(
                f"action: setup_outcoming | result: success | queue: {self._outcoming_data.outcome_queue}"
            )

            # This blocks until stop_consuming() is called
            self._middleware.start_consuming()

        except Exception as e:
            logging.error(f"Failed to initialize BlackBoxSession: {e}")
            raise
        finally:
            if self._middleware:
                self._middleware.close()
                logging.info("Middleware connection closed")

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
            result = self._incoming_data.process_data_batch(body)

            # Check if batch was duplicate (returns None)
            if result is None:
                logging.debug("Duplicate batch detected, skipping processing")
                # ACK the duplicate message to remove it from queue
                self._middleware.ack_message(ch, method.delivery_tag)
                return

            predictions, is_last_batch, batch_index, session_id = result

            # Send predictions to the server
            self._outcoming_data.send_predictions(
                predictions=predictions,
                is_last_batch=is_last_batch,
                batch_index=batch_index,
                session_id=session_id,
            )

            # ACK the message only after successfully sending the response
            self._middleware.ack_message(ch, method.delivery_tag)
            logging.debug(f"Successfully processed and ACKed batch_index={batch_index}")

        except ValueError as e:
            # Data parsing/validation errors - NACK without requeue
            logging.error(f"action: handle_data | result: parse_error | error: {e}")
            self._middleware.nack_message(ch, method.delivery_tag, requeue=False)

        except Exception as e:
            # Other errors - NACK with requeue for retry
            logging.error(f"action: handle_data | result: fail | error: {e}")
            self._middleware.nack_message(ch, method.delivery_tag, requeue=True)
