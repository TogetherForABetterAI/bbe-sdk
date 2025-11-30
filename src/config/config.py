import os


CONNECTION_SERVICE_BASE_URL = f"http://connection-service:8080"


def initialize_config():
    config_params = {}

    config_params["logging_level"] = os.getenv("LOGGING_LEVEL", "INFO")

    return config_params


config_params = initialize_config()
