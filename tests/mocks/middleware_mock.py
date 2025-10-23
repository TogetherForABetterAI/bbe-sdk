from unittest.mock import Mock, MagicMock
from typing import Callable, Any, List, Dict
import threading
import time


class MiddlewareMock:
    """Mock de RabbitMQ"""

    def __init__(self):
        self.callback_function: Callable = None
        self.is_started = False
        self.is_connected = False
        self.sent_messages: List[Dict[str, Any]] = []
        self.connection_error = False
        self.send_error = False
        self.mock_instance = None
        self._setup_mock()

    def _setup_mock(self):
        self.mock_instance = Mock()
        self.mock_instance.start = Mock(side_effect=self._start)
        self.mock_instance.basic_send = Mock(side_effect=self._basic_send)
        self.mock_instance.close = Mock(side_effect=self._close)
        self.mock_instance.is_connected = Mock(side_effect=lambda: self.is_connected)

        # exponer los métodos directamente en esta instancia
        self.start = self.mock_instance.start
        self.basic_send = self.mock_instance.basic_send
        self.close = self.mock_instance.close

    @classmethod
    def create_successful_middleware(cls, callback_function: Callable = None):
        """Mock del middleware que funciona correctamente"""
        middleware = cls()
        if callback_function:
            middleware.callback_function = callback_function
        middleware.is_connected = True
        middleware.is_started = True
        return middleware

    def _start(self):
        """Simula el inicio del middleware"""
        if self.connection_error:
            raise Exception("Failed to connect to middleware")

        self.is_started = True
        self.is_connected = True

    def _basic_send(self, message: bytes, exchange_name: str):
        """Simula un basic send"""
        if self.send_error:
            raise Exception("Failed to send message to exchange")

        if self.connection_error and not self.is_connected:
            raise Exception("Not connected to middleware")

        self.sent_messages.append(
            {
                "message": message,
                "exchange_name": exchange_name,
                "timestamp": time.time(),
            }
        )

    def _close(self):
        """Simula el cierre del middleware"""
        self.is_started = False
        self.is_connected = False

    def simulate_incoming_message(
        self, data_batch_bytes: bytes, batch_index: int = 0, is_last_batch: bool = False
    ):
        """Simula un mensaje entrante del middleware"""
        if not self.is_started or not self.callback_function:
            raise Exception("Middleware not started or no callback function set")

        ch = Mock()
        method = Mock()
        method.delivery_tag = f"delivery_tag_{batch_index}"

        properties = Mock()
        properties.headers = {
            "batch_index": batch_index,
            "is_last_batch": is_last_batch,
        }

        self.callback_function(ch, method, properties, data_batch_bytes)

    def simulate_multiple_messages(
        self, data_batches: List[bytes], start_batch_index: int = 0
    ):
        """Simula múltiples mensajes entrantes secuencialmente"""
        for i, batch_data in enumerate(data_batches):
            batch_index = start_batch_index + i
            is_last_batch = i == len(data_batches) - 1

            self.simulate_incoming_message(batch_data, batch_index, is_last_batch)

            time.sleep(0.001)

    def clear_sent_messages(self):
        """Limpia los mensajes enviados almacenados"""
        self.sent_messages.clear()

    def get_sent_messages(self) -> List[Dict[str, Any]]:
        """Retorna todos los mensajes enviados"""
        return self.sent_messages.copy()

    def get_last_sent_message(self) -> Dict[str, Any]:
        """Retorna el último mensaje enviado"""
        if not self.sent_messages:
            raise ValueError("No messages have been sent")
        return self.sent_messages[-1]


class MiddlewareFactory:
    """Factory para crear diferentes tipos de mocks de rabbit"""

    @staticmethod
    def patch_middleware_setup():
        """Parche para reemplazar la configuración del middleware"""
        from unittest.mock import patch

        return patch("bbe_sdk.middleware.middleware.Middleware.setup_client_middleware")

    @staticmethod
    def create_for_session_test(callback_function: Callable = None):
        return MiddlewareMock.create_successful_middleware(callback_function)
