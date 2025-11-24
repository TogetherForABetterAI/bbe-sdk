"""
Common utilities for client authentication and session management.

This module provides shared functionality for both MNIST and ACDC evaluation clients.
"""

import os
import time
import uuid
import requests
from typing import Optional

def get_auth_token(username: str) -> Optional[str]:
    """
    Get authentication token for a client.

    Args:
        username: The client identifier

    Returns:
        Authentication token if successful, None otherwise
    """
    try:
        print(f"Obtaining token for username: {username}...")
        response = requests.post(
            f"http://users-service:8000/tokens/create",
            json={"username": username},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        response.raise_for_status()
        token = response.json().get("token")
        if not token:
            raise ValueError("Response does not contain a valid token.")
        print("Token obtained successfully.")
        return token
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 500:
            print(f"Server error obtaining token. User might not exist.")
            return None
        print(f"Error obtaining authentication token: {e}")
        raise
    except Exception as e:
        print(f"Error obtaining authentication token: {e}")
        raise


def create_user(
    username: str, model_type: str, inputs_format: str, outputs_format: str
) -> str:
    """
    Create a new user in the system.

    Args:
        username: Username for the new user
        model_type: Type of model (e.g., "mnist", "acdc")
        inputs_format: Format string for input data (e.g., "(28, 28, 1)")
        outputs_format: Format string for output data (e.g., "(10,)")

    Returns:
        client_id of the created user

    Raises:
        Exception: If user creation fails
    """
    try:
        print(f"Creating user '{username}' with model_type='{model_type}'...")
        response = requests.post(
            f"http://users-service:8000/users/create",
            json={
                "username": username,
                "email": f"{username}@example.com",
                "model_type": model_type,
                "inputs_format": inputs_format,
                "outputs_format": outputs_format,
            },
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        response.raise_for_status()
        user_id = response.json().get("id")
        if not user_id:
            raise ValueError("Response does not contain a valid user_id.")
        print(f"User created successfully. user_id: {user_id}")
        return user_id
    except Exception as e:
        print(f"Error creating user: {e}")
        raise


def get_or_create_user(
    model_type: str,
    inputs_format: str,
    outputs_format: str,
    user_id_file: str = "user_id.txt",
) -> tuple[str, str]:
    """
    Create a new user with a fresh UUID-based username.

    This function always creates a new user to avoid conflicts with existing users.
    The username is generated using UUID to ensure uniqueness.

    Args:
        model_type: Type of model (e.g., "mnist", "acdc")
        inputs_format: Format string for input data
        outputs_format: Format string for output data
        user_id_file: Path to file storing user_id (kept for compatibility, will be overwritten)

    Returns:
        Tuple of (user_id, token)
    """
    print("=" * 60)
    print(f"Setting up NEW user for model_type: {model_type}")
    print("=" * 60)

    # Generate unique username using UUID
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    username = f"user_{model_type}_{unique_id}"

    print(f"Creating new user with username: {username}")

    # Create the user
    user_id = create_user(username, model_type, inputs_format, outputs_format)

    # Try to save user_id to file (for reference/debugging)
    try:
        with open(user_id_file, "w") as f:
            f.write(user_id)
        print(f"user_id saved to {user_id_file}")
    except OSError:
        print(f"Could not save user_id to file (read-only filesystem)")
        print(f"   User ID: {user_id}")

    # Get token for new user
    token = get_auth_token(username)

    if not token:
        raise ValueError("Failed to obtain valid authentication token.")

    print(f"\nAuthentication successful")
    print(f"  username: {username}")
    print(f"  client_id: {user_id}")
    print(f"  token: {token[:20]}...")
    print("=" * 60)

    return user_id, token


def create_blackbox_interface(model):
    """
    Create a blackbox evaluation interface for a model.

    Args:
        model: Model object with a predict() method that handles batches

    Returns:
        Callable that takes a batch of data and returns predictions
    """

    def eval_input_batch(batch_data):
        # Model.predict() already handles batches, so pass the whole batch
        predictions = model.predict(batch_data)
        return predictions

    return eval_input_batch
