import pytest
import responses
import numpy as np
import time
import threading
from unittest.mock import patch, Mock
from src.session import BlackBoxSession
from src.models.errors import InvalidTokenError
from tests.mocks.auth_server_mock import UsersServerMock
from tests.mocks.middleware_mock import MiddlewareFactory, MiddlewareMock
from tests.mocks.protobuf_fixtures import ProtobufFixtures
from tests.utils.test_helpers import IntegrationTestBase


class TestBlackBoxSessionEndToEnd(IntegrationTestBase):

    @responses.activate
    def test_complete_image_classification_flow(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Flujo completo de clasificación de imágenes"""

        # Definir función de evaluación personalizada
        def image_classifier(data):
            batch_size = data.shape[0]
            predictions = []
            for i in range(batch_size):
                probs = np.random.dirichlet(np.ones(10))
                predictions.append(probs.tolist())
            return predictions

        session, mock_helper = self.setup_successful_session(
            sample_user_data, valid_token, image_classifier
        )

        assert session._user_id == sample_user_data["user_id"]
        assert session._inputs_format.shape == (1, 28, 28)
        assert session._inputs_format.dtype == np.float32

        batch_sizes = [1, 3, 2, 5]

        for i, batch_size in enumerate(batch_sizes):
            is_last = i == len(batch_sizes) - 1

            image_batch = ProtobufFixtures.create_image_batch(
                batch_size=batch_size, batch_index=i, is_last_batch=is_last
            )

            session.get_data(
                None, None, None, ProtobufFixtures.serialize_data_batch(image_batch)
            )

        assert session._count == len(batch_sizes)
        assert len(mock_helper.sent_messages) == len(batch_sizes)

        for i, sent_message in enumerate(mock_helper.sent_messages):
            predictions = ProtobufFixtures.deserialize_predictions(
                sent_message["message"]
            )

            assert predictions.batch_index == i
            assert predictions.eof == (i == len(batch_sizes) - 1)

            expected_batch_size = batch_sizes[i]
            assert len(predictions.pred) == expected_batch_size

            for pred_list in predictions.pred:
                assert len(pred_list.values) == 10
                assert all(0.0 <= prob <= 1.0 for prob in pred_list.values)

    @responses.activate
    def test_complete_tabular_regression_flow(
        self, sample_user_data_tabular, valid_token, disable_logging
    ):
        """Test: Flujo completo de regresión con datos tabulares"""

        def regression_model(data):
            batch_size = data.shape[0]
            predictions = []
            for i in range(batch_size):
                pred_value = np.sum(data[i]) + np.random.normal(0, 0.1)
                predictions.append([pred_value])
            return predictions

        session, mock_helper = self.setup_successful_session(
            sample_user_data_tabular, valid_token, regression_model
        )

        assert session._inputs_format.shape == (10,)
        assert session._inputs_format.dtype == np.float32
        assert session._model_type == "regression"

        tabular_batch = ProtobufFixtures.create_tabular_batch(
            batch_size=4, features=10, is_last_batch=True
        )

        session.get_data(
            None, None, None, ProtobufFixtures.serialize_data_batch(tabular_batch)
        )

        assert len(mock_helper.sent_messages) == 1
        predictions = ProtobufFixtures.deserialize_predictions(
            mock_helper.sent_messages[0]["message"]
        )

        assert len(predictions.pred) == 4
        assert predictions.eof == True

        for pred_list in predictions.pred:
            assert len(pred_list.values) == 1

    @responses.activate
    def test_error_recovery_flow(self, sample_user_data, valid_token, disable_logging):
        """Test: Flujo con manejo de errores y recuperación"""

        call_count = 0

        def flaky_eval_function(data):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise Exception("Model temporarily unavailable")

            batch_size = data.shape[0]
            return [[0.5, 0.3, 0.2] + [0.0] * 7 for _ in range(batch_size)]

        session, mock_helper = self.setup_successful_session(
            sample_user_data, valid_token, flaky_eval_function
        )

        batch1 = ProtobufFixtures.create_image_batch(batch_size=1, batch_index=0)

        with pytest.raises(Exception, match="Model temporarily unavailable"):
            session.get_data(
                None, None, None, ProtobufFixtures.serialize_data_batch(batch1)
            )

        batch2 = ProtobufFixtures.create_image_batch(
            batch_size=2, batch_index=1, is_last_batch=True
        )

        session.get_data(
            None, None, None, ProtobufFixtures.serialize_data_batch(batch2)
        )

        assert len(mock_helper.sent_messages) == 1
        predictions = ProtobufFixtures.deserialize_predictions(
            mock_helper.sent_messages[0]["message"]
        )
        assert len(predictions.pred) == 2
        assert predictions.batch_index == 1
        assert predictions.eof == True

    @responses.activate
    def test_multiple_batch_sequence_flow(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Procesamiento de secuencia de múltiples batches"""

        def sequence_eval_function(data):
            batch_size = data.shape[0]
            return [[0.8, 0.1, 0.1] + [0.0] * 7 for _ in range(batch_size)]

        session, mock_helper = self.setup_successful_session(
            sample_user_data, valid_token, sequence_eval_function
        )

        num_batches = 5
        batch_sizes = [2, 3, 1, 4, 2]

        for i, batch_size in enumerate(batch_sizes):
            is_last = i == num_batches - 1

            batch = ProtobufFixtures.create_image_batch(
                batch_size=batch_size, batch_index=i, is_last_batch=is_last
            )

            session.get_data(
                None, None, None, ProtobufFixtures.serialize_data_batch(batch)
            )

        assert session._count == num_batches
        assert len(mock_helper.sent_messages) == num_batches

        for i, sent_message in enumerate(mock_helper.sent_messages):
            predictions = ProtobufFixtures.deserialize_predictions(
                sent_message["message"]
            )
            assert predictions.batch_index == i
            assert predictions.eof == (i == num_batches - 1)
            assert len(predictions.pred) == batch_sizes[i]

    @responses.activate
    def test_session_auto_start_integration(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Session starts middleware automatically on initialization"""

        def dummy_eval_function(data):
            return [[1.0] + [0.0] * 9 for _ in range(data.shape[0])]

        session, mock_helper = self.setup_successful_session(
            sample_user_data, valid_token, dummy_eval_function
        )

        # Middleware should be started automatically during init
        assert mock_helper.middleware_mock.is_started == True
        assert session._middleware is not None

    @responses.activate
    def test_session_initialization_with_middleware_error(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Session raises error if middleware fails to start automatically"""

        def dummy_eval_function(data):
            return [[1.0] + [0.0] * 9 for _ in range(data.shape[0])]

        # Setup auth mocks first
        UsersServerMock.setup_successful_auth(sample_user_data, valid_token)

        middleware_mock = Mock()
        middleware_mock.start = Mock(
            side_effect=Exception("Failed to connect to middleware")
        )
        middleware_mock.close = Mock()
        middleware_mock.is_started = False

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_setup.return_value = middleware_mock

            # Since middleware starts automatically during __init__,
            # the exception should be raised during initialization
            with pytest.raises(Exception, match="Failed to connect to middleware"):
                session = BlackBoxSession(
                    eval_input_batch=dummy_eval_function,
                    token=valid_token,
                    user_id=sample_user_data["user_id"],
                )

    @responses.activate
    def test_performance_with_large_batches(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Rendimiento con batches grandes"""

        def large_batch_eval_function(data):
            batch_size = data.shape[0]
            return np.random.dirichlet(np.ones(10), size=batch_size).tolist()

        session, mock_helper = self.setup_successful_session(
            sample_user_data, valid_token, large_batch_eval_function
        )

        large_batch = ProtobufFixtures.create_image_batch(
            batch_size=100, batch_index=0, is_last_batch=True
        )

        start_time = time.time()
        session.get_data(
            None, None, None, ProtobufFixtures.serialize_data_batch(large_batch)
        )
        processing_time = time.time() - start_time

        assert len(mock_helper.sent_messages) == 1
        predictions = ProtobufFixtures.deserialize_predictions(
            mock_helper.sent_messages[0]["message"]
        )

        assert len(predictions.pred) == 100
        assert predictions.eof == True

        assert processing_time < 5.0

    @responses.activate
    def test_concurrent_session_simulation(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Simulación de procesamiento concurrente"""

        def concurrent_eval_function(data):
            batch_size = data.shape[0]
            thread_id = threading.current_thread().ident
            return [
                [float(thread_id % 1000) / 1000.0] + [0.0] * 9
                for _ in range(batch_size)
            ]

        session, mock_helper = self.setup_successful_session(
            sample_user_data, valid_token, concurrent_eval_function
        )

        num_calls = 5

        for i in range(num_calls):
            batch = ProtobufFixtures.create_image_batch(
                batch_size=1, batch_index=i, is_last_batch=(i == num_calls - 1)
            )

            session.get_data(
                None, None, None, ProtobufFixtures.serialize_data_batch(batch)
            )

        assert len(mock_helper.sent_messages) == num_calls

        for i, sent_message in enumerate(mock_helper.sent_messages):
            predictions = ProtobufFixtures.deserialize_predictions(
                sent_message["message"]
            )
            assert predictions.batch_index == i
            assert len(predictions.pred) == 1
            assert 0.0 <= predictions.pred[0].values[0] <= 1.0

    @responses.activate
    def test_different_shapes_end_to_end(
        self, sample_user_data_different_shapes, valid_token, disable_logging
    ):
        """Test: Flujo end-to-end con diferentes shapes de entrada"""

        def flexible_eval_function(data):
            batch_size = data.shape[0]
            return [[0.8, 0.2] for _ in range(batch_size)]

        for user_data in sample_user_data_different_shapes:
            responses.reset()

            session, mock_helper = self.setup_successful_session(
                user_data, valid_token, flexible_eval_function
            )

            from src.config.data import parse_inputs_format

            expected_format = parse_inputs_format(user_data["inputs_format"])
            assert session._inputs_format.shape == expected_format.shape
            assert session._inputs_format.dtype == np.float32

            test_data = np.random.rand(2, *expected_format.shape).astype(np.float32)

            from src.proto import dataset_pb2

            data_batch = dataset_pb2.DataBatch()
            data_batch.data = test_data.tobytes()
            data_batch.batch_index = 0
            data_batch.is_last_batch = True

            session.get_data(None, None, None, data_batch.SerializeToString())

            assert len(mock_helper.sent_messages) == 1
            predictions = ProtobufFixtures.deserialize_predictions(
                mock_helper.sent_messages[0]["message"]
            )

            assert len(predictions.pred) == 2
            assert predictions.eof == True

            for pred_list in predictions.pred:
                assert len(pred_list.values) == 2
