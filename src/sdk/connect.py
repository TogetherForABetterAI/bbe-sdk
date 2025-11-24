"""
Connect component: Handles authentication, token validation, and user information retrieval.
"""

import logging
import requests
from typing import Optional, Callable
from src.config.config import CONNECTION_SERVICE_BASE_URL
from src.middleware.middleware import Middleware
from src.models.errors import InvalidTokenError
from src.models.responses import (
    TokenValidationResponse,
    RabbitMQCredentials,
    ConnectResponse,
)
from src.models.inputs_format import InputsFormat


class Connect:
    """
    Manages connection with the server and handles authentication.

    Responsibilities:
    - Token validation
    - User information retrieval
    - Service connection establishment
    - RabbitMQ credentials retrieval
    - Middleware creation and initialization

    The Connect component encapsulates all connection logic.
    Methods are public and should be called explicitly from the session.
    """

    def __init__(
        self,
        token: str,
        client_id: str,
        base_url: Optional[str] = None,  # For testing purposes
        http_client: Optional[Callable] = None,  # For testing purposes
        middleware_factory: Optional[Callable] = None,  # For testing purposes
    ):
        """
        Initialize the connection component.

        Args:
            token: Authentication token
            client_id: Client identifier
            base_url: Base URL for the users service (defaults to config value)
            http_client: HTTP client for requests (for testing, defaults to requests module)
            middleware_factory: Factory function to create middleware (for testing)
        """
        self.token = token
        self.client_id = client_id
        self.base_url = base_url or CONNECTION_SERVICE_BASE_URL
        self._http_client = http_client or requests
        self._middleware_factory = middleware_factory or Middleware
        self._rabbitmq_credentials = None
        self._middleware = None

    def connect_to_service(self) -> ConnectResponse:
        """
        Establish connection with the server and retrieve RabbitMQ credentials.

        Returns:
            ConnectResponse object with credentials and user configuration

        Raises:
            InvalidTokenError: If connection establishment fails
            RuntimeError: If credentials are missing in response
        """
        try:
            connect_resp = self._http_client.post(
                f"{self.base_url}/users/connect",
                json={"client_id": self.client_id, "token": self.token},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to server: {e}")

        if connect_resp.status_code != 200:
            raise InvalidTokenError(
                f"Connection service returned status {connect_resp.status_code}: {connect_resp.text}"
            )

        connect_data = connect_resp.json()
        if connect_data.get("status") != "success":
            raise InvalidTokenError(
                f"Connection failed: {connect_data.get('message', 'Unknown error')}"
            )

        # Extract RabbitMQ credentials from response
        credentials = connect_data.get("credentials")
        if not credentials:
            raise RuntimeError("Connection response missing RabbitMQ credentials")

        rabbitmq_credentials = RabbitMQCredentials(
            username=credentials.get("username"),
            password=credentials.get("password"),
            host=credentials.get("host"),
            port=credentials.get("port"),
        )

        # Store credentials for middleware creation
        self._rabbitmq_credentials = rabbitmq_credentials

        # Create ConnectResponse with all data
        response = ConnectResponse(
            status=connect_data.get("status"),
            message=connect_data.get("message"),
            credentials=rabbitmq_credentials,
            inputs_format=connect_data.get("inputs_format", ""),
        )

        logging.info(
            f"action: connect_to_service | result: success | client_id: {self.client_id} | "
            f"message: {response.message} | rabbitmq_host: {rabbitmq_credentials.host}"
        )

        return response

    def create_middleware(self):
        """
        Create and initialize middleware connection with RabbitMQ.

        Returns:
            Initialized Middleware instance

        Raises:
            RuntimeError: If middleware creation fails or credentials not set
        """
        if not self._rabbitmq_credentials:
            raise RuntimeError(
                "RabbitMQ credentials not available. Call connect_to_service() first."
            )

        try:
            self._middleware = self._middleware_factory(
                host=self._rabbitmq_credentials.host,
                port=self._rabbitmq_credentials.port,
                username=self._rabbitmq_credentials.username,
                password=self._rabbitmq_credentials.password,
            )
            logging.info(
                f"action: create_middleware | result: success | client_id: {self.client_id}"
            )
            return self._middleware
        except Exception as e:
            logging.error(f"action: create_middleware | result: fail | error: {e}")
            raise RuntimeError(f"Failed to create middleware connection: {e}")

    def try_connect(self) -> tuple:
        """
        Execute the complete connection flow and return middleware and connection response.

        This method orchestrates all connection steps:
        1. Validate token
        2. Connect to service and get RabbitMQ credentials + user config
        3. Create and return middleware instance

        Returns:
            Tuple of (middleware, ConnectResponse) ready to use

        Raises:
            InvalidTokenError: If authentication or connection fails
            RuntimeError: If any step in the connection process fails
        """
        logging.info(
            f"action: try_connect | status: starting | client_id: {self.client_id}"
        )

        connect_response = self.connect_to_service()
        middleware = self.create_middleware()

        logging.info(
            f"action: try_connect | status: success | client_id: {self.client_id}"
        )
        return middleware, connect_response.inputs_format
