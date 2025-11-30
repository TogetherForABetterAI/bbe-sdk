from src.pb.outcomingData import calibration_pb2
import pytest
import responses
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from src.session import BlackBoxSession
from src.proto import dataset_pb2
from tests.mocks.auth_server_mock import UsersServerMock
from tests.mocks.middleware_mock import MiddlewareFactory, MiddlewareMock
from tests.mocks.protobuf_fixtures import ProtobufFixtures
from tests.utils.test_helpers import IntegrationTestBase


class TestBlackBoxSessionDataProcessing(IntegrationTestBase):

    @responses.activate
    def test_get_data_successful_image_processing(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Procesamiento exitoso de datos de imagen"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        image_batch = ProtobufFixtures.create_image_batch(
            batch_size=2, height=28, width=28, channels=1
        )
        data_batch_bytes = ProtobufFixtures.serialize_data_batch(image_batch)

        session.get_data(None, None, None, data_batch_bytes)

        assert len(middleware_mock.sent_messages) == 1
        sent_message = middleware_mock.sent_messages[0]
        assert sent_message["exchange_name"] == "replies_exchange"

        predictions = ProtobufFixtures.deserialize_predictions(sent_message["message"])
        assert predictions.batch_index == image_batch.batch_index
        assert predictions.eof == image_batch.is_last_batch
        assert len(predictions.pred) == 2

    @responses.activate
    def test_get_data_successful_tabular_processing(
        self,
        sample_user_data_tabular,
        valid_token,
        mock_eval_function_regression,
        disable_logging,
    ):
        """Test: Procesamiento exitoso de datos tabulares"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data_tabular, valid_token, mock_eval_function_regression
        )

        tabular_batch = ProtobufFixtures.create_tabular_batch(batch_size=3, features=10)
        data_batch_bytes = ProtobufFixtures.serialize_data_batch(tabular_batch)

        session.get_data(None, None, None, data_batch_bytes)

        assert len(middleware_mock.sent_messages) == 1
        sent_message = middleware_mock.sent_messages[0]

        predictions = ProtobufFixtures.deserialize_predictions(sent_message["message"])
        assert len(predictions.pred) == 3

    @responses.activate
    def test_get_data_single_sample(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Procesamiento de una sola muestra"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        single_batch = ProtobufFixtures.create_image_batch(
            batch_size=1, height=28, width=28, channels=1, is_last_batch=True
        )
        data_batch_bytes = ProtobufFixtures.serialize_data_batch(single_batch)

        session.get_data(None, None, None, data_batch_bytes)

        assert len(middleware_mock.sent_messages) == 1
        predictions = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert len(predictions.pred) == 1
        assert predictions.eof == True

    @responses.activate
    def test_get_data_incompatible_data_size(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Error con formato de datos incompatible"""
        session, _ = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        incompatible_batch = ProtobufFixtures.create_incompatible_batch(
            batch_size=1, wrong_shape=(5, 5)
        )
        data_batch_bytes = ProtobufFixtures.serialize_data_batch(incompatible_batch)

        with pytest.raises(ValueError, match="Data size incompatible"):
            session.get_data(None, None, None, data_batch_bytes)

    @responses.activate
    def test_get_data_wrong_dtype(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Error con tipo de datos incorrecto"""
        session, _ = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        # Crear datos con bytes que no se pueden interpretar correctamente como float32
        # 21 bytes no es divisible por 4 (tamaño de float32)
        wrong_data = b"this_is_21_bytes_long"  # Exactamente 21 bytes
        data_batch = dataset_pb2.DataBatch()
        data_batch.data = wrong_data
        data_batch.batch_index = 0
        data_batch.is_last_batch = False
        data_batch_bytes = data_batch.SerializeToString()

        # Debe fallar al parsear con dtype incorrecto
        with pytest.raises(ValueError, match="Failed to parse data buffer"):
            session.get_data(None, None, None, data_batch_bytes)

    @responses.activate
    def test_get_data_empty_batch(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Manejo de batch vacío"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        empty_batch = ProtobufFixtures.create_empty_batch()
        data_batch_bytes = ProtobufFixtures.serialize_data_batch(empty_batch)

        session.get_data(None, None, None, data_batch_bytes)

        assert len(middleware_mock.sent_messages) == 1
        predictions = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert len(predictions.pred) == 0

    @responses.activate
    def test_get_data_eval_function_error(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Error en la función de evaluación del usuario"""

        def failing_eval_function(data):
            raise Exception("Model evaluation failed")

        session, _ = self.setup_successful_session(
            sample_user_data, valid_token, failing_eval_function
        )

        valid_batch = ProtobufFixtures.create_image_batch(batch_size=1)
        data_batch_bytes = ProtobufFixtures.serialize_data_batch(valid_batch)

        with pytest.raises(Exception, match="Model evaluation failed"):
            session.get_data(None, None, None, data_batch_bytes)

    @responses.activate
    def test_get_data_send_error_closes_middleware(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Error al enviar cierra el middleware"""
        UsersServerMock.setup_successful_auth(sample_user_data, valid_token)

        middleware_mock = MiddlewareMock()
        middleware_mock.send_error = True
        mock_middleware_instance = middleware_mock.mock_instance

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token=valid_token,
                user_id=sample_user_data["user_id"],
            )

            valid_batch = ProtobufFixtures.create_image_batch(batch_size=1)
            data_batch_bytes = ProtobufFixtures.serialize_data_batch(valid_batch)

            with pytest.raises(Exception, match="Failed to send message"):
                session.get_data(None, None, None, data_batch_bytes)

            mock_middleware_instance.close.assert_called_once()

    @responses.activate
    def test_get_data_malformed_protobuf(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Error con protobuf malformado"""
        session, _ = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        invalid_protobuf_data = b"invalid_protobuf_data"

        with pytest.raises(Exception):
            session.get_data(None, None, None, invalid_protobuf_data)

    @responses.activate
    def test_get_data_increments_counter(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Verificar que se incrementa el contador de mensajes"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        initial_count = session._count

        batch1 = ProtobufFixtures.create_image_batch(batch_size=1, batch_index=0)
        session.get_data(
            None, None, None, ProtobufFixtures.serialize_data_batch(batch1)
        )
        assert session._count == initial_count + 1

        batch2 = ProtobufFixtures.create_image_batch(batch_size=1, batch_index=1)
        session.get_data(
            None, None, None, ProtobufFixtures.serialize_data_batch(batch2)
        )
        assert session._count == initial_count + 2

    @responses.activate
    def test_get_data_different_batch_sizes(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Procesamiento con diferentes tamaños de batch"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        batch_sizes = [1, 3, 5, 10]

        for batch_size in batch_sizes:
            middleware_mock.sent_messages.clear()

            batch = ProtobufFixtures.create_image_batch(batch_size=batch_size)
            data_batch_bytes = ProtobufFixtures.serialize_data_batch(batch)

            session.get_data(None, None, None, data_batch_bytes)

            predictions = ProtobufFixtures.deserialize_predictions(
                middleware_mock.sent_messages[0]["message"]
            )
            assert len(predictions.pred) == batch_size

    @responses.activate
    def test_get_data_preserves_batch_metadata(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Verificar que se preservan los metadatos del batch"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        batch_index = 42
        is_last_batch = True

        batch = ProtobufFixtures.create_image_batch(
            batch_size=2, batch_index=batch_index, is_last_batch=is_last_batch
        )
        data_batch_bytes = ProtobufFixtures.serialize_data_batch(batch)

        session.get_data(None, None, None, data_batch_bytes)

        predictions = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert predictions.batch_index == batch_index
        assert predictions.eof == is_last_batch
