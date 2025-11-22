"""
Connect component: Handles authentication, token validation, and user information retrieval.
"""

import logging
import requests
from src.utils.data import parse_inputs_format


class InvalidTokenError(Exception):
    """Exception raised when token validation fails."""

    pass


class Connect:
    """
    Manages connection to the users-service and handles authentication.

    Responsibilities:
    - Token validation
    - User information retrieval
    - Service connection establishment
    """

    def __init__(self, token: str, client_id: str):
        """
        Initialize the connection component.

        Args:
            token: Authentication token
            client_id: Client identifier

        Raises:
            InvalidTokenError: If token validation or connection fails
        """
        self.token = token
        self.client_id = client_id
        self._user_info = None

        # Validate token and retrieve user information
        self._validate_token()
        self._retrieve_user_info()
        self._connect_to_service()

    def _validate_token(self) -> None:
        """
        Validate the authentication token with the users-service.

        Raises:
            InvalidTokenError: If token validation fails
        """
        try:
            validate_token_resp = requests.post(
                "http://users-service:8000/tokens/validate",
                json={"token": self.token, "client_id": self.client_id},
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to users server: {e}")

        if validate_token_resp.status_code != 200:
            raise InvalidTokenError(
                f"Users server returned status {validate_token_resp.status_code}: {validate_token_resp.text}"
            )

        validate_token_data = validate_token_resp.json()
        if not validate_token_data.get("is_valid", False):
            raise InvalidTokenError("Token validation failed: not authorized.")

    def _retrieve_user_info(self) -> None:
        """
        Retrieve user information from the users-service.

        Raises:
            InvalidTokenError: If user info retrieval fails
            RuntimeError: If input format specification is invalid
        """
        try:
            user_info_resp = requests.get(
                f"http://users-service:8000/users/{self.client_id}",
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to users server: {e}")

        if user_info_resp.status_code != 200:
            raise InvalidTokenError(
                f"Users server returned status {user_info_resp.status_code}: {user_info_resp.text}"
            )

        self._user_info = user_info_resp.json()

        # Parse and validate inputs format
        inputs_format_str = self._user_info.get("inputs_format", "")
        inputs_format = parse_inputs_format(inputs_format_str)

        if not inputs_format:
            raise RuntimeError(
                f"System configuration error: Invalid input format specification '{inputs_format_str}' for user {self.client_id}."
            )

        self._user_info["parsed_inputs_format"] = inputs_format

        logging.debug(
            f"action: receive_user_info | result: success | User info: "
            f"{self._user_info.get('client_id')}, {self._user_info.get('username')}, "
            f"{self._user_info.get('email')}, {self._user_info.get('model_type')}, "
            f"{inputs_format}, {self._user_info.get('outputs_format')}"
        )

    def _connect_to_service(self) -> None:
        """
        Establish connection to the users-service.

        Raises:
            InvalidTokenError: If connection establishment fails
        """
        try:
            connect_resp = requests.post(
                "http://users-service:8000/users/connect",
                json={"client_id": self.client_id, "token": self.token},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as e:
            raise InvalidTokenError(f"Failed to connect to users-service: {e}")

        if connect_resp.status_code != 200:
            raise InvalidTokenError(
                f"Connection service returned status {connect_resp.status_code}: {connect_resp.text}"
            )

        connect_data = connect_resp.json()
        if connect_data.get("status") != "success":
            raise InvalidTokenError(
                f"Connection failed: {connect_data.get('message', 'Unknown error')}"
            )

        logging.info(
            f"action: connect_to_service | result: success | client_id: {self.client_id}"
        )

    @property
    def user_info(self) -> dict:
        """Get user information dictionary."""
        return self._user_info

    @property
    def inputs_format(self):
        """Get parsed inputs format."""
        return self._user_info.get("parsed_inputs_format")

    @property
    def username(self) -> str:
        """Get username."""
        return self._user_info.get("username", "")

    @property
    def email(self) -> str:
        """Get user email."""
        return self._user_info.get("email", "")

    @property
    def model_type(self) -> str:
        """Get model type."""
        return self._user_info.get("model_type", "")

    @property
    def outputs_format(self) -> str:
        """Get outputs format."""
        return self._user_info.get("outputs_format", "")
