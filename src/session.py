import logging
from .sdk.connect import Connect
from .sdk.incoming_data import IncomingData
from .sdk.outcoming_data import OutcomingData
from .models.inputs_format import parse_inputs_format
import numpy as np


class BlackBoxSession:
    def __init__(self, eval_input_batch, token, user_id):
        # 2. Use an Adapter to inject user_id into every log automatically
        logger = logging.getLogger(__name__)
        self.logger = logging.LoggerAdapter(logger, {'user_id': user_id, 'component': 'Session'})
        
        self.logger.info("Initializing BlackBoxSession")
        
        self._middleware = None
        self._user_id = user_id
        self._inputs_format = None
        self._incoming_data = None
        self._outcoming_data = None

        self._build_state(eval_input_batch, token, user_id)

    def _build_state(self, eval_input_batch, token, user_id):
        try:
            connect = Connect(token=token, user_id=user_id, logger=self.logger)

            middleware, inputs_format = connect.try_connect()

            inputs_format = parse_inputs_format(inputs_format)
            if not inputs_format:

                self.logger.error("Invalid input format detected", extra={
                    'raw_format': inputs_format
                })
                raise RuntimeError(
                    f"System configuration error: Invalid input format specification for user {user_id}."
                )

            self._user_id = user_id
            self._inputs_format = inputs_format
            self._middleware = middleware

            self._incoming_data = IncomingData(
                inputs_format=self._inputs_format,
                on_message_callback=eval_input_batch,
                user_id=user_id,
                logger=self.logger
            )

            self._middleware.basic_consume(
                self._incoming_data.income_queue, self._handle_incoming_data
            )
            
            self.logger.info("Consumption setup successful", extra={
                'queue': self._incoming_data.income_queue,
                'action': 'setup_consumption'
            })

            self._outcoming_data = OutcomingData(
                middleware=self._middleware, user_id=user_id, logger=self.logger
            )
            
            self.logger.info("Outcoming data channel ready", extra={
                'queue': self._outcoming_data.outcome_queue,
                'action': 'setup_outcoming'
            })

            self.logger.info("Blocking to consume messages...")
            self._middleware.start_consuming()

        except Exception as e:
            self.logger.exception("Critical failure initializing BlackBoxSession")
            raise
        finally:
            if self._middleware:
                self._middleware.close()
                self.logger.info("Middleware connection closed")

    def _handle_incoming_data(self, ch, method, properties, body):        
        try:
            result = self._incoming_data.process_data_batch(body)

            if result is None:
                self.logger.warning("Duplicate batch detected", extra={
                    'action': 'duplicate_skip'
                })
                self._middleware.ack_message(ch, method.delivery_tag)
                return

            predictions, is_last_batch, batch_index, session_id = result

            self.logger.debug("Batch processed locally", extra={
                'batch_index': batch_index,
                'session_id': session_id,
                'prediction_count': len(predictions) if hasattr(predictions, '__len__') else 0
            })

            self._outcoming_data.send_predictions(
                predictions=predictions,
                is_last_batch=is_last_batch,
                batch_index=batch_index,
                session_id=session_id,
            )

            self._middleware.ack_message(ch, method.delivery_tag)

        
        except ValueError as e:
            self.logger.error("Data validation failed", exc_info=True, extra={
                'error_type': 'ValueError',
                'delivery_tag': method.delivery_tag
            })
            self._middleware.nack_message(ch, method.delivery_tag, requeue=False)

        except Exception as e:
            self.logger.exception("Unexpected error in message loop", extra={
                'delivery_tag': method.delivery_tag,
                'retry': True
            })
            self._middleware.nack_message(ch, method.delivery_tag, requeue=True)