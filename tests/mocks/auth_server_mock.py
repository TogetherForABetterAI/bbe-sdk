import responses
import json
from typing import Dict, Any, Optional


class UsersServerMock:
    """Mock del servidor de autenticación para tests"""

    BASE_URL = "http://users-service:8000"

    @classmethod
    def setup_successful_auth(
        cls, user_data: Dict[str, Any], token: str = "valid_token"
    ):
        """Mock para autenticación exitosa"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            json={"is_valid": True},
            status=200,
        )

        responses.add(
            responses.GET,
            f"{cls.BASE_URL}/users/{user_data['client_id']}",
            json=user_data,
            status=200,
        )

        # Mock users-service /users/connect endpoint
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/users/connect",
            json={"status": "success", "message": "Connected successfully"},
            status=200,
        )

    @classmethod
    def setup_invalid_token(cls, client_id: str = "test_client"):
        """Mock para token inválido"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            json={"is_valid": False, "error": "Invalid token"},
            status=401,
        )

    @classmethod
    def setup_token_validation_connection_error(cls):
        """Mock para error de conexión en validación de token"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            body=ConnectionError("Connection failed to users server"),
        )

    @classmethod
    def setup_user_info_connection_error(cls, client_id: str = "test_client"):
        """Mock para error de conexión en obtención de info de usuario"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            json={"is_valid": True},
            status=200,
        )

        responses.add(
            responses.GET,
            f"{cls.BASE_URL}/users/{client_id}",
            body=ConnectionError("Connection failed to users server"),
        )

    @classmethod
    def setup_token_validation_server_error(cls):
        """Mock para error del servidor en validación de token"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            json={"error": "Internal server error"},
            status=500,
        )

    @classmethod
    def setup_user_info_server_error(cls, client_id: str = "test_client"):
        """Mock para error del servidor en obtención de info de usuario"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            json={"is_valid": True},
            status=200,
        )

        responses.add(
            responses.GET,
            f"{cls.BASE_URL}/users/{client_id}",
            json={"error": "User not found"},
            status=404,
        )

    @classmethod
    def setup_user_with_invalid_format(cls, client_id: str = "test_client"):
        """Mock para usuario con formato de entrada inválido"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            json={"is_valid": True},
            status=200,
        )

        user_data = {
            "client_id": client_id,
            "username": "testuser",
            "email": "test@example.com",
            "model_type": "classification",
            "inputs_format": "",  # Formato vacío/inválido
            "outputs_format": "(10,)",
        }

        responses.add(
            responses.GET,
            f"{cls.BASE_URL}/users/{client_id}",
            json=user_data,
            status=200,
        )

    @classmethod
    def setup_user_with_missing_format(cls, client_id: str = "test_client"):
        """Mock para usuario sin inputs format"""
        responses.add(
            responses.POST,
            f"{cls.BASE_URL}/tokens/validate",
            json={"is_valid": True},
            status=200,
        )

        user_data = {
            "client_id": client_id,
            "username": "testuser",
            "email": "test@example.com",
            "model_type": "classification",
            # Sin inputs_format
            "outputs_format": "(10,)",
        }

        responses.add(
            responses.GET,
            f"{cls.BASE_URL}/users/{client_id}",
            json=user_data,
            status=200,
        )

    @classmethod
    def setup_timeout_error(cls, endpoint: str = "both"):
        """Mock para timeout"""
        from requests.exceptions import Timeout

        if endpoint in ["token", "both"]:
            responses.add(
                responses.POST,
                f"{cls.BASE_URL}/tokens/validate",
                body=Timeout("Request timed out"),
            )

        if endpoint in ["user", "both"]:
            # Si solo user endpoint falla, necesitamos token validation exitosa
            if endpoint == "user":
                responses.add(
                    responses.POST,
                    f"{cls.BASE_URL}/tokens/validate",
                    json={"is_valid": True},
                    status=200,
                )

            responses.add(
                responses.GET,
                f"{cls.BASE_URL}/users/test_client",
                body=Timeout("Request timed out"),
            )

    @classmethod
    def clear_all_mocks(cls):
        """Limpia todos los mocks de responses"""
        responses.reset()
        responses.stop()

    @classmethod
    def get_request_history(cls):
        """Retorna el historial de requests realizados"""
        return responses.calls
