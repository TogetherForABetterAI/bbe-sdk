"""
Integration tests for ACDC cardiac segmentation model evaluation.

These tests verify the end-to-end functionality of the BBE SDK
with ACDC-specific data formats and segmentation predictions.
"""

import pytest
import responses
import numpy as np
from unittest.mock import patch, Mock
from src.session import BlackBoxSession
from src.models.errors import InvalidTokenError
from tests.mocks.auth_server_mock import UsersServerMock
from tests.mocks.middleware_mock import MiddlewareFactory


class TestACDCSessionInitialization:
    """Test suite for ACDC model session initialization"""

    @responses.activate
    def test_successful_acdc_initialization(
        self,
        sample_user_data_acdc,
        valid_token,
        mock_eval_function_acdc,
        disable_logging,
    ):
        """Test: Successful initialization with ACDC model configuration"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function_acdc,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Verify session initialization
            assert session._user_id == sample_user_data_acdc["user_id"]
            assert session._username == sample_user_data_acdc["username"]
            assert session._model_type == sample_user_data_acdc["model_type"]

            # Verify input format for ACDC (256x256 grayscale images)
            assert session._inputs_format is not None
            assert session._inputs_format.shape == (256, 256, 1)
            assert session._inputs_format.dtype == np.float32

            # Verify middleware was set up correctly
            mock_middleware_setup.assert_called_once_with(
                user_id=sample_user_data_acdc["user_id"],
                callback_function=session.get_data,
            )

            assert session._on_message == mock_eval_function_acdc
            assert session._count == 0

    @responses.activate
    def test_acdc_invalid_token(self, mock_eval_function_acdc, disable_logging):
        """Test: ACDC session initialization with invalid token"""
        UsersServerMock.setup_invalid_token()

        with pytest.raises(InvalidTokenError, match="Users server returned status 401"):
            BlackBoxSession(
                eval_input_batch=mock_eval_function_acdc,
                token="invalid_token",
                user_id="test_acdc_client",
            )


class TestACDCDataProcessing:
    """Test suite for ACDC data processing"""

    @responses.activate
    def test_process_acdc_batch(
        self,
        sample_user_data_acdc,
        valid_token,
        mock_eval_function_acdc,
        sample_data_batch_acdc,
        disable_logging,
    ):
        """Test: Process a batch of ACDC cardiac MRI images"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function_acdc,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Process the data batch
            session.get_data(
                None, None, None, sample_data_batch_acdc.SerializeToString()
            )

            # Verify that predictions were sent
            assert len(mock_middleware_instance.sent_messages) == 1
            sent_message = mock_middleware_instance.sent_messages[0]
            assert "message" in sent_message
            assert sent_message["exchange_name"] == "replies_exchange"

    @responses.activate
    def test_acdc_batch_shape_validation(
        self,
        sample_user_data_acdc,
        valid_token,
        mock_eval_function_acdc,
        disable_logging,
    ):
        """Test: Verify correct shape handling for ACDC images"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            # Create session
            session = BlackBoxSession(
                eval_input_batch=mock_eval_function_acdc,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Create a batch with correct ACDC dimensions
            batch_size = 5
            correct_data = np.random.rand(batch_size, 256, 256, 1).astype(np.float32)

            from src.pb.dataset_service import dataset_service_pb2

            data_batch = dataset_service_pb2.DataBatchUnlabeled()
            data_batch.data = correct_data.tobytes()
            data_batch.batch_index = 0
            data_batch.is_last_batch = False

            # Process the batch - should work without errors
            session.get_data(None, None, None, data_batch.SerializeToString())

            # Verify processing succeeded
            assert len(mock_middleware_instance.sent_messages) == 1

    @responses.activate
    def test_acdc_predictions_format(
        self,
        sample_user_data_acdc,
        valid_token,
        sample_data_batch_acdc,
        disable_logging,
    ):
        """Test: Verify ACDC predictions have correct format (4 class probabilities)"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        # Create a mock function that returns proper ACDC predictions
        def acdc_eval_with_validation(data):
            batch_size = data.shape[0]
            predictions = []
            for i in range(batch_size):
                # Each prediction should have 4 probabilities (4 segmentation classes)
                probs = np.random.dirichlet(np.ones(4))  # Ensures sum to 1
                predictions.append(probs.tolist())
            return predictions

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=acdc_eval_with_validation,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Process batch
            session.get_data(
                None, None, None, sample_data_batch_acdc.SerializeToString()
            )

            # Verify message was sent
            assert len(mock_middleware_instance.sent_messages) == 1


class TestACDCEndToEnd:
    """End-to-end tests for ACDC workflow"""

    @responses.activate
    def test_complete_acdc_evaluation_flow(
        self,
        sample_user_data_acdc,
        valid_token,
        mock_eval_function_acdc,
        disable_logging,
    ):
        """Test: Complete ACDC evaluation flow with multiple batches"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function_acdc,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Simulate processing multiple batches
            from src.pb.dataset_service import dataset_service_pb2

            num_batches = 3
            batch_size = 10

            for batch_idx in range(num_batches):
                # Create batch
                batch_data = np.random.rand(batch_size, 256, 256, 1).astype(np.float32)
                data_batch = dataset_service_pb2.DataBatchUnlabeled()
                data_batch.data = batch_data.tobytes()
                data_batch.batch_index = batch_idx
                data_batch.is_last_batch = batch_idx == num_batches - 1

                # Process batch
                session.get_data(None, None, None, data_batch.SerializeToString())

            # Verify all batches were processed
            assert len(mock_middleware_instance.sent_messages) == num_batches

            # Verify last batch has correct EOF flag
            # (Would need to deserialize to check, this is simplified)
            assert (
                mock_middleware_instance.sent_messages[-1]["exchange_name"]
                == "replies_exchange"
            )

    @responses.activate
    def test_acdc_with_actual_segmenter_interface(
        self, sample_user_data_acdc, valid_token, disable_logging
    ):
        """Test: ACDC evaluation with realistic segmenter interface"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        # Create a mock segmenter that mimics ACDCSegmenter.predict() behavior
        class MockACDCSegmenter:
            def predict(self, images):
                """
                Simulates ACDCSegmenter.predict() behavior.
                Returns list of probability distributions for each image.
                """
                batch_size = images.shape[0]
                predictions = []
                for i in range(batch_size):
                    # Simulate per-pixel probabilities averaged to class probabilities
                    # Returns shape (4,) for 4 classes
                    avg_probs = np.random.dirichlet(np.ones(4)).tolist()
                    predictions.append(avg_probs)
                return predictions

        segmenter = MockACDCSegmenter()

        # Create the evaluation interface (as done in main_acdc.py)
        def eval_input_batch(batch_data):
            return segmenter.predict(batch_data)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=eval_input_batch,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Create and process a batch
            from src.pb.dataset_service import dataset_service_pb2

            batch_data = np.random.rand(5, 256, 256, 1).astype(np.float32)
            data_batch = dataset_service_pb2.DataBatchUnlabeled()
            data_batch.data = batch_data.tobytes()
            data_batch.batch_index = 0
            data_batch.is_last_batch = True

            # Process
            session.get_data(None, None, None, data_batch.SerializeToString())

            # Verify successful processing
            assert len(mock_middleware_instance.sent_messages) == 1


class TestACDCErrorHandling:
    """Test error handling specific to ACDC models"""

    @responses.activate
    def test_acdc_incompatible_batch_size(
        self,
        sample_user_data_acdc,
        valid_token,
        mock_eval_function_acdc,
        disable_logging,
    ):
        """Test: Handle incompatible batch size for ACDC images"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=mock_eval_function_acdc,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Create batch with wrong total size (not divisible by 256*256*1)
            from src.pb.dataset_service import dataset_service_pb2

            wrong_size_data = np.random.rand(100).astype(np.float32)  # Wrong size
            data_batch = dataset_service_pb2.DataBatchUnlabeled()
            data_batch.data = wrong_size_data.tobytes()
            data_batch.batch_index = 0
            data_batch.is_last_batch = False

            # Should raise ValueError due to incompatible size
            with pytest.raises(ValueError, match="Data size incompatible"):
                session.get_data(None, None, None, data_batch.SerializeToString())

    @responses.activate
    def test_acdc_model_inference_error(
        self,
        sample_user_data_acdc,
        valid_token,
        sample_data_batch_acdc,
        disable_logging,
    ):
        """Test: Handle model inference errors gracefully"""
        UsersServerMock.setup_successful_auth(sample_user_data_acdc, valid_token)

        # Create a function that raises an error during inference
        def failing_eval(data):
            raise RuntimeError("Model inference failed")

        with MiddlewareFactory.patch_middleware_setup() as mock_middleware_setup:
            mock_middleware_instance = MiddlewareFactory.create_for_session_test()
            mock_middleware_setup.return_value = mock_middleware_instance

            session = BlackBoxSession(
                eval_input_batch=failing_eval,
                token=valid_token,
                user_id=sample_user_data_acdc["user_id"],
            )

            # Should raise the inference error
            with pytest.raises(RuntimeError, match="Model inference failed"):
                session.get_data(
                    None, None, None, sample_data_batch_acdc.SerializeToString()
                )
