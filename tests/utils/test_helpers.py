import numpy as np
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock
from bbe_sdk.utils.data import InputsFormat
from bbe_sdk.session import BlackBoxSession
from tests.mocks.auth_server_mock import UsersServerMock
from tests.mocks.middleware_mock import MiddlewareFactory


class IntegrationTestBase:
    def setup_successful_session(self, user_data, token, eval_function):
        """Helper para crear una sesión exitosa"""
        UsersServerMock.setup_successful_auth(user_data, token)

        self.middleware_mock = Mock()
        self.sent_messages = []

        def mock_basic_send(message, exchange_name):
            self.sent_messages.append(
                {"message": message, "exchange_name": exchange_name}
            )

        self.middleware_mock.basic_send = Mock(side_effect=mock_basic_send)
        self.middleware_mock.start = Mock()
        self.middleware_mock.close = Mock()

        self.middleware_mock.is_started = True
        self.middleware_mock.is_connected = True

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_setup.return_value = self.middleware_mock

            session = BlackBoxSession(
                eval_input_batch=eval_function,
                token=token,
                client_id=user_data["client_id"],
            )

            return session, self


class TestHelpers:
    @staticmethod
    def create_test_user_data(
        client_id: str = "test_client",
        model_type: str = "classification",
        inputs_format: str = "image:float32:(1,28,28)",
        outputs_format: str = "float32:(10,)",
    ) -> Dict[str, Any]:
        """Creamos datos de usuario para tests"""
        return {
            "client_id": client_id,
            "username": f"user_{client_id}",
            "email": f"{client_id}@test.com",
            "model_type": model_type,
            "inputs_format": inputs_format,
            "outputs_format": outputs_format,
        }
