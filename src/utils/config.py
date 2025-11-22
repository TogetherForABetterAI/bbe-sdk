import os

DATASET_EXCHANGE = "dataset_exchange"
REPLIES_EXCHANGE = "replies_exchange"


def initialize_config():
    config_params = {}

    config_params["logging_level"] = os.getenv("LOGGING_LEVEL", "INFO")

    config_params["rabbitmq_host"] = os.getenv("RABBITMQ_HOST", "rabbitmq")
    config_params["rabbitmq_port"] = int(os.getenv("RABBITMQ_PORT", "5672"))

    return config_params


config_params = initialize_config()
