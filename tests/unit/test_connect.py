"""
Unit tests for Connect component.

These tests demonstrate how the refactored Connect class is testeable
using mocks and dependency injection.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.sdk.connect import Connect
from src.models.errors import InvalidTokenError


class TestConnect:
    """Test suite for Connect component."""

    def setup_method(self):
        """Setup test fixtures."""
        self.token = "test_token_123"
        self.client_id = "test_client_456"
        self.base_url = "http://test-service:8000"

        # Mock HTTP client
        self.mock_http_client = Mock()

        # Mock middleware factory
        self.mock_middleware_factory = Mock()
        self.mock_middleware = Mock()
        self.mock_middleware_factory.return_value = self.mock_middleware

    def test_validate_token_success(self):
        """Test successful token validation."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_valid": True}
        self.mock_http_client.post.return_value = mock_response

        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
        )

        # Act
        connect.validate_token()

        # Assert
        self.mock_http_client.post.assert_called_once_with(
            f"{self.base_url}/tokens/validate",
            json={"token": self.token, "client_id": self.client_id},
            timeout=5,
        )

    def test_validate_token_invalid(self):
        """Test token validation with invalid token."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_valid": False}
        self.mock_http_client.post.return_value = mock_response

        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
        )

        # Act & Assert
        with pytest.raises(InvalidTokenError, match="Token validation failed"):
            connect.validate_token()

    def test_validate_token_server_error(self):
        """Test token validation with server error."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        self.mock_http_client.post.return_value = mock_response

        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
        )

        # Act & Assert
        with pytest.raises(InvalidTokenError, match="Users server returned status 500"):
            connect.validate_token()

    def test_retrieve_user_info_success(self):
        """Test successful user info retrieval."""
        # Arrange
        user_data = {
            "client_id": self.client_id,
            "username": "test_user",
            "email": "test@example.com",
            "model_type": "classification",
            "inputs_format": "float32,(28,28,1)",
            "outputs_format": "float32,(10,)",
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = user_data
        self.mock_http_client.get.return_value = mock_response

        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
        )

        # Act
        result = connect.retrieve_user_info()

        # Assert
        assert result["username"] == "test_user"
        assert result["email"] == "test@example.com"
        assert "parsed_inputs_format" in result
        self.mock_http_client.get.assert_called_once()

    def test_connect_to_service_success(self):
        """Test successful connection to service."""
        # Arrange
        credentials = {
            "username": "rabbitmq_user",
            "password": "secret_pass",
            "host": "rabbitmq.example.com",
            "port": 5672,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "message": "Connected successfully",
            "credentials": credentials,
        }
        self.mock_http_client.post.return_value = mock_response

        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
        )

        # Act
        result = connect.connect_to_service()

        # Assert
        assert result["username"] == "rabbitmq_user"
        assert result["password"] == "secret_pass"
        assert result["host"] == "rabbitmq.example.com"
        assert result["port"] == 5672

    def test_connect_to_service_missing_credentials(self):
        """Test connection when credentials are missing."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "message": "Connected",
            "credentials": None,  # Missing credentials
        }
        self.mock_http_client.post.return_value = mock_response

        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="missing RabbitMQ credentials"):
            connect.connect_to_service()

    def test_create_middleware_success(self):
        """Test successful middleware creation."""
        # Arrange
        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
            middleware_factory=self.mock_middleware_factory,
        )

        # Set credentials manually (normally done by connect_to_service)
        connect._rabbitmq_credentials = {
            "username": "user",
            "password": "pass",
            "host": "localhost",
            "port": 5672,
        }

        # Act
        result = connect.create_middleware()

        # Assert
        assert result == self.mock_middleware
        self.mock_middleware_factory.assert_called_once_with(
            host="localhost",
            port=5672,
            username="user",
            password="pass",
            routing_key=self.client_id,
        )

    def test_create_middleware_without_credentials(self):
        """Test middleware creation without credentials fails."""
        # Arrange
        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
            middleware_factory=self.mock_middleware_factory,
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="RabbitMQ credentials not available"):
            connect.create_middleware()

    def test_full_connection_flow(self):
        """Test the complete connection flow."""
        # Arrange
        # Mock validate_token
        mock_validate_resp = Mock()
        mock_validate_resp.status_code = 200
        mock_validate_resp.json.return_value = {"is_valid": True}

        # Mock retrieve_user_info
        mock_user_resp = Mock()
        mock_user_resp.status_code = 200
        mock_user_resp.json.return_value = {
            "client_id": self.client_id,
            "username": "test_user",
            "email": "test@example.com",
            "model_type": "classification",
            "inputs_format": "float32,(28,28,1)",
            "outputs_format": "float32,(10,)",
        }

        # Mock connect_to_service
        mock_connect_resp = Mock()
        mock_connect_resp.status_code = 200
        mock_connect_resp.json.return_value = {
            "status": "success",
            "message": "Connected",
            "credentials": {
                "username": "rmq_user",
                "password": "rmq_pass",
                "host": "rabbitmq.test",
                "port": 5672,
            },
        }

        # Setup mock responses
        self.mock_http_client.post.side_effect = [mock_validate_resp, mock_connect_resp]
        self.mock_http_client.get.return_value = mock_user_resp

        connect = Connect(
            token=self.token,
            client_id=self.client_id,
            base_url=self.base_url,
            http_client=self.mock_http_client,
            middleware_factory=self.mock_middleware_factory,
        )

        # Act
        connect.validate_token()
        user_info = connect.retrieve_user_info()
        rabbitmq_creds = connect.connect_to_service()
        middleware = connect.create_middleware()

        # Assert
        assert user_info["username"] == "test_user"
        assert rabbitmq_creds["host"] == "rabbitmq.test"
        assert middleware == self.mock_middleware
        assert connect.get_username() == "test_user"
        assert connect.get_email() == "test@example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
