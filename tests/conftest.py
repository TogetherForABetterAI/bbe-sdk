import pytest
import numpy as np
from unittest.mock import Mock, patch
import responses
from src.config.data import InputsFormat
from src.pb.dataset_service import dataset_service_pb2
from src.pb.outcomingData import calibration_pb2


@pytest.fixture
def sample_user_data():
    """Mock para mnist"""
    return {
        "client_id": "test_client_123",
        "username": "testuser",
        "email": "test@example.com",
        "model_type": "classification",
        "inputs_format": "(1,28,28)",
        "outputs_format": "(10,)",
    }


@pytest.fixture
def sample_user_data_tabular():
    """Mock para datos tabulares"""
    return {
        "client_id": "test_client_456",
        "username": "testuser2",
        "email": "test2@example.com",
        "model_type": "regression",
        "inputs_format": "(10,)",
        "outputs_format": "(1,)",
    }


@pytest.fixture
def sample_user_data_acdc():
    """Mock para ACDC cardiac segmentation"""
    return {
        "client_id": "test_client_acdc_789",
        "username": "testuser_acdc",
        "email": "test_acdc@example.com",
        "model_type": "acdc",
        "inputs_format": "(256,256,1)",
        "outputs_format": "(4,)",
    }


@pytest.fixture
def sample_input_format():
    """Formato de entrada mock para imágenes"""
    return InputsFormat(dtype=np.float32, shape=(1, 28, 28))


@pytest.fixture
def sample_input_format_tabular():
    """Formato de entrada mock para datos tabulares"""
    return InputsFormat(dtype=np.float32, shape=(10,))


@pytest.fixture
def sample_input_format_acdc():
    """Formato de entrada mock para imágenes ACDC"""
    return InputsFormat(dtype=np.float32, shape=(256, 256, 1))


@pytest.fixture
def sample_user_data_different_shapes():
    """Lista de datos de usuario con diferentes shapes para testing"""
    return [
        {
            "client_id": "test_client_rgb",
            "username": "testuser_rgb",
            "model_type": "classification",
            "inputs_format": "(3,224,224)",  # RGB image
            "outputs_format": "(1000,)",
        },
        {
            "client_id": "test_client_grayscale",
            "username": "testuser_gray",
            "model_type": "classification",
            "inputs_format": "(1,256,256)",  # Grayscale image
            "outputs_format": "(5,)",
        },
        {
            "client_id": "test_client_3d",
            "username": "testuser_3d",
            "model_type": "regression",
            "inputs_format": "(16,16,16)",  # 3D data
            "outputs_format": "(1,)",
        },
        {
            "client_id": "test_client_vector",
            "username": "testuser_vector",
            "model_type": "classification",
            "inputs_format": "(784,)",  # Flattened 28x28
            "outputs_format": "(10,)",
        },
    ]


@pytest.fixture
def mock_eval_function():
    """Función de evaluación mock que retorna predicciones para clasificación"""

    def eval_fn(data):
        batch_size = data.shape[0]
        # Retorna 10 probabilidades por muestra
        return np.random.rand(batch_size, 10).tolist()

    return eval_fn


@pytest.fixture
def mock_eval_function_regression():
    """Función de evaluación mock que retorna predicciones para regresión"""

    def eval_fn(data):
        batch_size = data.shape[0]
        # Retorna 1 valor por muestra
        return np.random.rand(batch_size, 1).tolist()

    return eval_fn


@pytest.fixture
def mock_eval_function_acdc():
    """Función de evaluación mock que retorna predicciones para ACDC segmentation"""

    def eval_fn(data):
        batch_size = data.shape[0]
        # Retorna 4 probabilidades por muestra (4 clases de segmentación)
        return np.random.rand(batch_size, 4).tolist()

    return eval_fn


@pytest.fixture
def sample_data_batch():
    """DataBatchUnlabeled protobuf mock para imágenes 28x28"""
    # Crear datos de muestra (2 imágenes 1x28x28)
    sample_data = np.random.rand(2, 1, 28, 28).astype(np.float32)

    data_batch = dataset_service_pb2.DataBatchUnlabeled()
    data_batch.data = sample_data.tobytes()
    data_batch.batch_index = 0
    data_batch.is_last_batch = False

    return data_batch


@pytest.fixture
def sample_data_batch_single():
    """DataBatchUnlabeled protobuf mock para una sola imagen 28x28"""
    # Crear datos de muestra (1 imagen 1x28x28)
    sample_data = np.random.rand(1, 1, 28, 28).astype(np.float32)

    data_batch = dataset_service_pb2.DataBatchUnlabeled()
    data_batch.data = sample_data.tobytes()
    data_batch.batch_index = 0
    data_batch.is_last_batch = True

    return data_batch


@pytest.fixture
def sample_data_batch_tabular():
    """DataBatchUnlabeled protobuf mock para datos tabulares"""
    # Crear datos de muestra (3 muestras de 10 features)
    sample_data = np.random.rand(3, 10).astype(np.float32)

    data_batch = dataset_service_pb2.DataBatchUnlabeled()
    data_batch.data = sample_data.tobytes()
    data_batch.batch_index = 1
    data_batch.is_last_batch = False

    return data_batch


@pytest.fixture
def invalid_data_batch():
    """DataBatchUnlabeled con datos incompatibles (tamaño incorrecto)"""
    # Crear datos incompatibles (5x5 en lugar de 28x28)
    invalid_data = np.random.rand(1, 5, 5).astype(np.float32)

    data_batch = dataset_service_pb2.DataBatchUnlabeled()
    data_batch.data = invalid_data.tobytes()
    data_batch.batch_index = 0
    data_batch.is_last_batch = False

    return data_batch


@pytest.fixture
def sample_data_batch_acdc():
    """DataBatchUnlabeled protobuf mock para imágenes ACDC 256x256"""
    # Crear datos de muestra (10 imágenes 256x256x1)
    sample_data = np.random.rand(10, 256, 256, 1).astype(np.float32)

    data_batch = dataset_service_pb2.DataBatchUnlabeled()
    data_batch.data = sample_data.tobytes()
    data_batch.batch_index = 0
    data_batch.is_last_batch = False

    return data_batch


@pytest.fixture
def valid_token():
    """Token válido para tests"""
    return "valid_test_token_123"


@pytest.fixture
def invalid_token():
    """Token inválido para tests"""
    return "invalid_token_456"


@pytest.fixture
def sample_predictions():
    """Predicciones mock para clasificación"""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]


@pytest.fixture
def sample_predictions_regression():
    """Predicciones mock para regresión"""
    return [[0.5], [1.2], [0.8]]


@pytest.fixture
def sample_predictions_acdc():
    """Predicciones mock para ACDC segmentation (4 clases)"""
    return [
        [0.6, 0.2, 0.15, 0.05],  # Background dominant
        [0.1, 0.7, 0.15, 0.05],  # RV dominant
        [0.1, 0.2, 0.65, 0.05],  # Myocardium dominant
        [0.1, 0.15, 0.1, 0.65],  # LV dominant
    ]


@pytest.fixture
def empty_predictions():
    """Lista vacía de predicciones"""
    return []


@pytest.fixture(autouse=True)
def reset_responses():
    """Fixture que limpia los mocks de responses después de cada test"""
    yield
    responses.reset()
    responses.stop()


@pytest.fixture
def disable_logging():
    """Desactiva logging durante los tests para output más limpio"""
    import logging

    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)
