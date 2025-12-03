import os


CONNECTION_SERVICE_BASE_URL = f"http://136.114.87.151"
DISPATCHER_QUEUE = "%s_dispatcher_queue"
CALIBRATON_QUEUE = "%s_outputs_cal_queue"


def initialize_config():
    config_params = {}

    config_params["logging_level"] = os.getenv("LOGGING_LEVEL", "INFO")

    return config_params


config_params = initialize_config()
