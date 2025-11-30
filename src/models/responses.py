"""
Response models from the Server.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class RabbitMQCredentials:
    """RabbitMQ connection credentials."""

    username: str
    password: str
    host: str
    port: int


@dataclass
class TokenValidationResponse:
    """Response from token validation endpoint."""

    is_valid: bool
    message: Optional[str] = None


@dataclass
class ConnectResponse:
    """Response from the connect endpoint."""

    status: str
    message: str
    credentials: RabbitMQCredentials
    inputs_format: str
