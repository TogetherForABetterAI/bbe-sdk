import pytest
import responses
import numpy as np
from unittest.mock import patch, Mock
from src.session import BlackBoxSession
from src.models.errors import InvalidTokenError
from tests.mocks.auth_server_mock import UsersServerMock
from tests.mocks.middleware_mock import MiddlewareFactory


class TestBlackBoxSessionInitialization:
    @responses.activate
    def test_successful_initialization(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):

        UsersServerMock.setup_successful_auth(sample_user_data, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token=valid_token,
                user_id=sample_user_data["user_id"],
            )

            assert session._user_id == sample_user_data["user_id"]
            assert session._username == sample_user_data["username"]
            assert session._email == sample_user_data["email"]
            assert session._model_type == sample_user_data["model_type"]

            assert session._inputs_format is not None
            assert session._inputs_format.shape == (1, 28, 28)
            assert session._inputs_format.dtype == np.float32

            mock_middleware_setup.assert_called_once_with(
                user_id=sample_user_data["user_id"],
                callback_function=session.get_data,
            )

            assert session._on_message == mock_eval_function
            assert session._count == 0

    @responses.activate
    def test_initialization_with_tabular_data(
        self,
        sample_user_data_tabular,
        valid_token,
        mock_eval_function_regression,
        disable_logging,
    ):
        """Test: Inicialización exitosa con datos tabulares"""
        UsersServerMock.setup_successful_auth(sample_user_data_tabular, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function_regression,
                token=valid_token,
                user_id=sample_user_data_tabular["user_id"],
            )

            assert session._inputs_format.shape == (10,)
            assert session._inputs_format.dtype == np.float32
            assert session._model_type == "regression"

    @responses.activate
    def test_invalid_token_error(self, mock_eval_function, disable_logging):
        """Test: Error con token inválido"""
        UsersServerMock.setup_invalid_token()

        with pytest.raises(InvalidTokenError, match="Users server returned status 401"):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="invalid_token",
                user_id="test_client",
            )

    @responses.activate
    def test_token_validation_connection_error(
        self, mock_eval_function, disable_logging
    ):
        """Test: Error de conexión al validar token"""
        UsersServerMock.setup_token_validation_connection_error()

        with pytest.raises(
            InvalidTokenError, match="Failed to connect to users server"
        ):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id="test_client",
            )

    @responses.activate
    def test_user_info_connection_error(self, mock_eval_function, disable_logging):
        """Test: Error de conexión al obtener info de usuario"""
        user_id = "test_client"
        UsersServerMock.setup_user_info_connection_error(user_id)

        with pytest.raises(
            InvalidTokenError, match="Failed to connect to users server"
        ):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id=user_id,
            )

    @responses.activate
    def test_token_validation_server_error(self, mock_eval_function, disable_logging):
        """Test: Error del servidor al validar token"""
        UsersServerMock.setup_token_validation_server_error()

        with pytest.raises(InvalidTokenError, match="Users server returned status 500"):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id="test_client",
            )

    @responses.activate
    def test_user_info_server_error(self, mock_eval_function, disable_logging):
        """Test: Error del servidor al obtener info de usuario"""
        user_id = "test_client"
        UsersServerMock.setup_user_info_server_error(user_id)

        with pytest.raises(InvalidTokenError, match="Users server returned status 404"):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id=user_id,
            )

    @responses.activate
    def test_invalid_input_format_empty(self, mock_eval_function, disable_logging):
        """Test: Error con formato de entrada vacío"""
        user_id = "test_client"
        UsersServerMock.setup_user_with_invalid_format(user_id)

        with pytest.raises(
            RuntimeError,
            match="System configuration error: Invalid input format specification",
        ):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id=user_id,
            )

    @responses.activate
    def test_missing_input_format(self, mock_eval_function, disable_logging):
        """Test: Error con formato de entrada faltante"""
        user_id = "test_client"
        UsersServerMock.setup_user_with_missing_format(user_id)

        with pytest.raises(
            RuntimeError,
            match="System configuration error: Invalid input format specification",
        ):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id=user_id,
            )

    @responses.activate
    def test_middleware_setup_error(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Error al configurar el middleware"""
        UsersServerMock.setup_successful_auth(sample_user_data, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_setup.side_effect = Exception("Failed to setup middleware")

            with pytest.raises(Exception, match="Failed to setup middleware"):
                BlackBoxSession(
                    eval_input_batch=mock_eval_function,
                    token=valid_token,
                    user_id=sample_user_data["user_id"],
                )

    @responses.activate
    def test_timeout_on_token_validation(self, mock_eval_function, disable_logging):
        """Test: Timeout en validación de token"""
        UsersServerMock.setup_timeout_error("token")

        with pytest.raises(
            InvalidTokenError, match="Failed to connect to users server"
        ):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id="test_client",
            )

    @responses.activate
    def test_timeout_on_user_info(self, mock_eval_function, disable_logging):
        """Test: Timeout en obtención de info de usuario"""
        UsersServerMock.setup_timeout_error("user")

        with pytest.raises(
            InvalidTokenError, match="Failed to connect to users server"
        ):
            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token="valid_token",
                user_id="test_client",
            )

    @responses.activate
    def test_partial_user_data(self, valid_token, mock_eval_function, disable_logging):
        """Test: Datos de usuario incompletos pero válidos"""
        partial_user_data = {
            "user_id": "test_client_partial",
            "username": "",
            "email": "",
            "model_type": "classification",
            "inputs_format": "(1,28,28)",
            "outputs_format": "(10,)",
        }

        UsersServerMock.setup_successful_auth(partial_user_data, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token=valid_token,
                user_id=partial_user_data["user_id"],
            )

            assert session._user_id == partial_user_data["user_id"]
            assert session._username == ""
            assert session._email == ""
            assert session._inputs_format is not None

    @responses.activate
    def test_request_history_validation(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Verificar que se realizan las llamadas HTTP correctas"""
        UsersServerMock.setup_successful_auth(sample_user_data, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token=valid_token,
                user_id=sample_user_data["user_id"],
            )

            # Expects 3 calls: token validation, user info, and users-service connect
            assert len(responses.calls) == 3

            token_call = responses.calls[0]
            assert token_call.request.method == "POST"
            assert "/tokens/validate" in token_call.request.url

            user_call = responses.calls[1]
            assert user_call.request.method == "GET"
            assert f"/users/{sample_user_data['user_id']}" in user_call.request.url

            connect_call = responses.calls[2]
            assert connect_call.request.method == "POST"
            assert "/users/connect" in connect_call.request.url

    @responses.activate
    def test_different_input_shapes(
        self,
        sample_user_data_different_shapes,
        valid_token,
        mock_eval_function,
        disable_logging,
    ):
        """Test: Inicialización con diferentes shapes de entrada"""

        for user_data in sample_user_data_different_shapes:
            responses.reset()

            UsersServerMock.setup_successful_auth(user_data, valid_token)

            with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
                mock_middleware_instance = MiddlewareFactory.create_for_session_test()
                mock_middleware_setup.return_value = mock_middleware_instance

                session = BlackBoxSession(
                    eval_input_batch=mock_eval_function,
                    token=valid_token,
                    user_id=user_data["user_id"],
                )

                from src.config.data import parse_inputs_format

                expected_format = parse_inputs_format(user_data["inputs_format"])
                assert session._inputs_format.shape == expected_format.shape
                assert session._inputs_format.dtype == np.float32
                assert session._user_id == user_data["user_id"]
                assert session._model_type == user_data["model_type"]

    @responses.activate
    def test_invalid_shape_formats(
        self, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Manejo de formatos de shape inválidos"""

        invalid_formats = [
            "(invalid,shape)",
            "()",
            "1,2,3",
            "(1.5,2,3)",
            "(1,-2,3)",
            "not_a_shape",
        ]

        for invalid_format in invalid_formats:
            responses.reset()

            user_data = {
                "user_id": "test_client_invalid",
                "username": "testuser",
                "email": "test@example.com",
                "model_type": "classification",
                "inputs_format": invalid_format,
                "outputs_format": "(10,)",
            }

            UsersServerMock.setup_successful_auth(user_data, valid_token)

            with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
                mock_middleware_instance = MiddlewareFactory.create_for_session_test()
                mock_middleware_setup.return_value = mock_middleware_instance

                with pytest.raises(ValueError, match="Invalid shape format"):
                    BlackBoxSession(
                        eval_input_batch=mock_eval_function,
                        token=valid_token,
                        user_id=user_data["user_id"],
                    )

    @responses.activate
    def test_edge_case_shapes(self, valid_token, mock_eval_function, disable_logging):
        """Test: Casos extremos de shapes válidos"""

        edge_case_shapes = [
            "(1,)",
            "(1,1)",
            "(1000,)",
            "(1,1,1,1,1)",
            "(512,512)",
        ]

        for shape_str in edge_case_shapes:
            responses.reset()

            user_data = {
                "user_id": f"test_client_{shape_str.replace(',', '_').replace('(', '').replace(')', '')}",
                "username": "testuser",
                "email": "test@example.com",
                "model_type": "classification",
                "inputs_format": shape_str,
                "outputs_format": "(2,)",
            }

            UsersServerMock.setup_successful_auth(user_data, valid_token)

            with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
                mock_middleware_instance = MiddlewareFactory.create_for_session_test()
                mock_middleware_setup.return_value = mock_middleware_instance

                session = BlackBoxSession(
                    eval_input_batch=mock_eval_function,
                    token=valid_token,
                    user_id=user_data["user_id"],
                )

                from src.config.data import parse_inputs_format

                expected_format = parse_inputs_format(shape_str)
                assert session._inputs_format.shape == expected_format.shape
                assert session._inputs_format.dtype == np.float32
