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
    RabbitMQCredentials,
    ConnectResponse,
)


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
            user_id: str,
            base_url: Optional[str] = None,
            http_client: Optional[Callable] = None,
            middleware_factory: Optional[Callable] = None,
            logger: Optional[logging.Logger] = None,  # <--- 1. INJECT LOGGER
        ):
        """
        Initialize the connection component.

        Args:
            token: Authentication token
            user_id: User identifier
            base_url: Base URL for the users service (defaults to config value)
            http_client: HTTP client for requests (for testing, defaults to requests module)
            middleware_factory: Factory function to create middleware (for testing)
        """
        self.token = token
        self.user_id = user_id
        self.base_url = base_url or CONNECTION_SERVICE_BASE_URL
        self._http_client = http_client or requests
        self._middleware_factory = middleware_factory or Middleware
        self._rabbitmq_credentials = None
        self._middleware = None

        base_logger = logger or logging.getLogger(__name__)
        self.logger = logging.LoggerAdapter(
            base_logger, 
            {'user_id': user_id, 'component': 'Connect'}
        )   

    def connect_to_service(self) -> ConnectResponse:
        """
        Establish connection with the server and retrieve RabbitMQ credentials.

        Returns:
            ConnectResponse object with credentials and user configuration

        Raises:
            InvalidTokenError: If connection establishment fails
            RuntimeError: If credentials are missing in response
        """
        url = f"{self.base_url}/sessions/start"
        try:
            connect_resp = self._http_client.post(
                url,
                json={"user_id": self.user_id, "token": self.token},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as e:
            self.logger.error("Network error connecting to auth service", exc_info=True)
            raise InvalidTokenError(f"Failed to connect to server: {e}")

        if connect_resp.status_code != 200:
            self.logger.error("Auth service returned non-200 status", extra={
                'status_code': connect_resp.status_code,
                'response_text': connect_resp.text
            })
            raise InvalidTokenError(
                f"Connection service returned status {connect_resp.status_code}: {connect_resp.text}"
            )

        connect_data = connect_resp.json()
        if connect_data.get("status") != "success":
            self.logger.warning("Connection refused by logic", extra={
                'api_message': connect_data.get('message')
            })
            raise InvalidTokenError(
                f"Connection failed: {connect_data.get('message', 'Unknown error')}"
            )

        # Extract RabbitMQ credentials from response
        credentials = connect_data.get("credentials")
        if not credentials:
            self.logger.critical("Protocol Error: Response missing credentials", extra={
                'response_keys': list(connect_data.keys())
            })
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

        self.logger.info("Service authentication successful", extra={
                    'rabbitmq_host': rabbitmq_credentials.host,
                    'api_message': response.message
                })
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
            self.logger.info("Middleware initialized", extra={
                            'host': self._rabbitmq_credentials.host
                        })
            return self._middleware
        except Exception as e:
            self.logger.exception("Failed to initialize Middleware connection")
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
        self.logger.debug("Starting connection flow")

        connect_response = self.connect_to_service()
        middleware = self.create_middleware()

        self.logger.info("Connection flow completed successfully")
        return middleware, connect_response.inputs_format
