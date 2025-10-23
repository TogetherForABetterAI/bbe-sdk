import pytest
import responses
import numpy as np
from unittest.mock import patch, Mock
from bbe_sdk.session import BlackBoxSession
from bbe_sdk.proto import calibration_pb2
from tests.mocks.auth_server_mock import UsersServerMock
from tests.mocks.middleware_mock import MiddlewareFactory, MiddlewareMock
from tests.mocks.protobuf_fixtures import ProtobufFixtures
from tests.utils.test_helpers import IntegrationTestBase


class TestBlackBoxSessionSendProbs(IntegrationTestBase):

    @responses.activate
    def test_send_probs_classification_successful(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Envío exitoso de predicciones de clasificación"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        predictions = [
            [0.1, 0.2, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025],
        ]

        session._send_probs(predictions, is_last_batch=True, batch_index=1)

        assert len(middleware_mock.sent_messages) == 1
        sent_message = middleware_mock.sent_messages[0]
        assert sent_message["exchange_name"] == "replies_exchange"

        pred_proto = ProtobufFixtures.deserialize_predictions(sent_message["message"])
        assert pred_proto.eof == True
        assert pred_proto.batch_index == 1
        assert len(pred_proto.pred) == 2

        for i, pred_list in enumerate(pred_proto.pred):
            assert len(pred_list.values) == 10
            for j, prob in enumerate(pred_list.values):
                assert abs(prob - predictions[i][j]) < 1e-6

    @responses.activate
    def test_send_probs_regression_successful(
        self,
        sample_user_data_tabular,
        valid_token,
        mock_eval_function_regression,
        disable_logging,
    ):
        """Test: Envío exitoso de predicciones de regresión"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data_tabular, valid_token, mock_eval_function_regression
        )

        predictions = [[0.5], [1.2], [0.8]]

        session._send_probs(predictions, is_last_batch=False, batch_index=2)

        assert len(middleware_mock.sent_messages) == 1
        sent_message = middleware_mock.sent_messages[0]

        pred_proto = ProtobufFixtures.deserialize_predictions(sent_message["message"])
        assert pred_proto.eof == False
        assert pred_proto.batch_index == 2
        assert len(pred_proto.pred) == 3

        for i, pred_list in enumerate(pred_proto.pred):
            assert len(pred_list.values) == 1
            assert abs(pred_list.values[0] - predictions[i][0]) < 1e-6

    @responses.activate
    def test_send_probs_numpy_array_input(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Envío con predicciones como numpy array"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        predictions_array = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.1, 0.1, 0.2, 0.05, 0.025, 0.025, 0.0, 0.0, 0.0],
            ]
        )

        session._send_probs(predictions_array, is_last_batch=True, batch_index=0)

        assert len(middleware_mock.sent_messages) == 1
        pred_proto = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert len(pred_proto.pred) == 2

        for i, pred_list in enumerate(pred_proto.pred):
            for j, prob in enumerate(pred_list.values):
                assert abs(prob - predictions_array[i][j]) < 1e-6

    @responses.activate
    def test_send_probs_single_prediction(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Envío de una sola predicción"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        single_prediction = [[0.9, 0.05, 0.025, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        session._send_probs(single_prediction, is_last_batch=True, batch_index=5)

        pred_proto = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert len(pred_proto.pred) == 1
        assert pred_proto.batch_index == 5
        assert pred_proto.eof == True

    @responses.activate
    def test_send_probs_empty_predictions(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Envío de predicciones vacías"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        empty_predictions = []

        session._send_probs(empty_predictions, is_last_batch=True, batch_index=0)

        pred_proto = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert len(pred_proto.pred) == 0
        assert pred_proto.eof == True

    @responses.activate
    def test_send_probs_middleware_send_error(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Error al enviar a través del middleware"""
        UsersServerMock.setup_successful_auth(sample_user_data, valid_token)

        middleware_mock = MiddlewareMock()
        middleware_mock.send_error = True
        mock_middleware_instance = middleware_mock.mock_instance

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function,
                token=valid_token,
                client_id=sample_user_data["client_id"],
            )

            predictions = [[0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

            with pytest.raises(Exception, match="Failed to send message"):
                session._send_probs(predictions, is_last_batch=True, batch_index=0)

    @responses.activate
    def test_send_probs_large_batch(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Envío de batch grande de predicciones"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        large_batch_size = 100
        large_predictions = []
        for i in range(large_batch_size):
            pred = [0.0] * 10
            pred[i % 10] = 0.8
            pred[(i + 1) % 10] = 0.2
            large_predictions.append(pred)

        session._send_probs(large_predictions, is_last_batch=False, batch_index=10)

        pred_proto = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert len(pred_proto.pred) == large_batch_size
        assert pred_proto.batch_index == 10
        assert pred_proto.eof == False

    @responses.activate
    def test_send_probs_variable_prediction_lengths(
        self, sample_user_data, valid_token, disable_logging
    ):
        """Test: Predicciones con diferentes longitudes, casos borde"""

        def variable_length_eval(data):
            return [[0.5, 0.3, 0.2], [0.8, 0.2], [1.0]]

        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, variable_length_eval
        )

        predictions = [[0.5, 0.3, 0.2], [0.8, 0.2], [1.0]]

        session._send_probs(predictions, is_last_batch=True, batch_index=0)

        pred_proto = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert len(pred_proto.pred) == 3
        assert len(pred_proto.pred[0].values) == 3
        assert len(pred_proto.pred[1].values) == 2
        assert len(pred_proto.pred[2].values) == 1

    @responses.activate
    def test_send_probs_negative_values(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Predicciones con valores negativos"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        predictions_with_negatives = [
            [-0.5, 0.3, 1.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        session._send_probs(
            predictions_with_negatives, is_last_batch=True, batch_index=3
        )

        pred_proto = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )

        assert abs(pred_proto.pred[0].values[0] - (-0.5)) < 1e-6
        assert abs(pred_proto.pred[1].values[1] - (-0.2)) < 1e-6

    @responses.activate
    def test_send_probs_multiple_calls_same_session(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Múltiples llamadas a send_probs en la misma sesión"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        predictions1 = [[0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        session._send_probs(predictions1, is_last_batch=False, batch_index=0)

        predictions2 = [[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        session._send_probs(predictions2, is_last_batch=True, batch_index=1)

        assert len(middleware_mock.sent_messages) == 2

        pred1 = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[0]["message"]
        )
        assert pred1.batch_index == 0
        assert pred1.eof == False

        pred2 = ProtobufFixtures.deserialize_predictions(
            middleware_mock.sent_messages[1]["message"]
        )
        assert pred2.batch_index == 1
        assert pred2.eof == True

    @responses.activate
    def test_send_probs_preserves_message_order(
        self, sample_user_data, valid_token, mock_eval_function, disable_logging
    ):
        """Test: Verificar que el orden de los mensajes se preserva"""
        session, middleware_mock = self.setup_successful_session(
            sample_user_data, valid_token, mock_eval_function
        )

        batch_indices = [5, 2, 8, 1, 3]

        for batch_idx in batch_indices:
            predictions = [[float(batch_idx)] + [0.0] * 9]
            session._send_probs(predictions, is_last_batch=False, batch_index=batch_idx)

        assert len(middleware_mock.sent_messages) == len(batch_indices)

        for i, expected_batch_idx in enumerate(batch_indices):
            pred_proto = ProtobufFixtures.deserialize_predictions(
                middleware_mock.sent_messages[i]["message"]
            )
            assert pred_proto.batch_index == expected_batch_idx
            assert pred_proto.pred[0].values[0] == float(expected_batch_idx)
