import numpy as np
from typing import List, Tuple, Optional
from bbe_sdk.proto import dataset_pb2, calibration_pb2


class ProtobufFixtures:
    """Fixtures para crear objetos protobuf de prueba"""

    @staticmethod
    def create_data_batch(
        data: np.ndarray, batch_index: int = 0, is_last_batch: bool = False
    ) -> dataset_pb2.DataBatch:
        """Crea un DataBatch protobuf con los datos especificados"""
        data_batch = dataset_pb2.DataBatch()
        data_batch.data = data.tobytes()
        data_batch.batch_index = batch_index
        data_batch.is_last_batch = is_last_batch
        return data_batch

    @staticmethod
    def create_image_batch(
        batch_size: int = 2,
        height: int = 28,
        width: int = 28,
        channels: int = 1,
        dtype: np.dtype = np.float32,
        batch_index: int = 0,
        is_last_batch: bool = False,
    ) -> dataset_pb2.DataBatch:
        """Crea un DataBatch con datos de imágenes"""

        if channels == 1:
            shape = (batch_size, channels, height, width)
        else:
            shape = (batch_size, height, width, channels)

        image_data = np.random.rand(*shape).astype(dtype)
        return ProtobufFixtures.create_data_batch(
            image_data, batch_index, is_last_batch
        )

    @staticmethod
    def create_tabular_batch(
        batch_size: int = 3,
        features: int = 10,
        dtype: np.dtype = np.float32,
        batch_index: int = 0,
        is_last_batch: bool = False,
    ) -> dataset_pb2.DataBatch:
        """Crea un DataBatch con datos tabulares"""
        tabular_data = np.random.rand(batch_size, features).astype(dtype)
        return ProtobufFixtures.create_data_batch(
            tabular_data, batch_index, is_last_batch
        )

    @staticmethod
    def create_incompatible_batch(
        batch_size: int = 1,
        wrong_shape: Tuple[int, ...] = (5, 5),
        dtype: np.dtype = np.float32,
    ) -> dataset_pb2.DataBatch:
        """Crea un DataBatch con datos incompatibles (para tests de error)"""
        incompatible_data = np.random.rand(batch_size, *wrong_shape).astype(dtype)
        return ProtobufFixtures.create_data_batch(incompatible_data, 0, False)

    @staticmethod
    def create_empty_batch() -> dataset_pb2.DataBatch:
        """Crea un DataBatch vacío"""
        data_batch = dataset_pb2.DataBatch()
        data_batch.data = b""
        data_batch.batch_index = 0
        data_batch.is_last_batch = True
        return data_batch

    @staticmethod
    def create_predictions(
        predictions: List[List[float]],
        batch_index: int = 0,
        is_last_batch: bool = False,
    ) -> calibration_pb2.Predictions:
        """Crea un objeto Predictions protobuf"""
        pred = calibration_pb2.Predictions()

        for prediction_list in predictions:
            pred_list = calibration_pb2.PredictionList()
            for prob in prediction_list:
                pred_list.values.append(prob)
            pred.pred.append(pred_list)

        pred.eof = is_last_batch
        pred.batch_index = batch_index
        return pred

    @staticmethod
    def serialize_data_batch(data_batch: dataset_pb2.DataBatch) -> bytes:
        """Serializa un DataBatch a bytes"""
        return data_batch.SerializeToString()

    @staticmethod
    def serialize_predictions(predictions: calibration_pb2.Predictions) -> bytes:
        """Serializa Predictions a bytes"""
        return predictions.SerializeToString()

    @staticmethod
    def deserialize_predictions(data: bytes) -> calibration_pb2.Predictions:
        """Deserializa bytes a Predictions"""
        predictions = calibration_pb2.Predictions()
        predictions.ParseFromString(data)
        return predictions
