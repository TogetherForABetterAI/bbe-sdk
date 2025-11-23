"""
Main entry point for MNIST digit classification model evaluation.

This script sets up the MNIST classification model and connects it to the
Black Box Evaluation (BBE) system for calibration assessment.
"""

import torch
from bbe_sdk import BlackBoxSession
from example_model.image_classifier import ImageClassifier
from utils.auth import get_or_create_user, create_blackbox_interface


def main():
    print("\n" + "=" * 60)
    print("MNIST Digit Classification Model - BBE Client")
    print("=" * 60 + "\n")

    # Model configuration
    MODEL_TYPE = "mnist"
    INPUTS_FORMAT = "(28, 28, 1)"  # Grayscale 28x28 images
    OUTPUTS_FORMAT = "(10,)"  # 10 digit classes
    MODEL_PATH = "example_model/model_state.pt"

    print("Initializing MNIST classification model...")
    print(f"  Model type: {MODEL_TYPE}")
    print(f"  Input format: {INPUTS_FORMAT}")
    print(f"  Output format: {OUTPUTS_FORMAT}")
    print(f"  Model weights: {MODEL_PATH}")
    print()

    # Initialize model
    model = ImageClassifier()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print("Model loaded successfully.\n")
    except FileNotFoundError:
        print(f"❌ Error: Model weights not found at '{MODEL_PATH}'")
        print("   Please ensure the model state file is in the correct location.")
        return

    # Get or create user and obtain authentication token
    try:
        user_id, token = get_or_create_user(
            model_type=MODEL_TYPE,
            inputs_format=INPUTS_FORMAT,
            outputs_format=OUTPUTS_FORMAT,
            user_id_file="user_id_mnist.txt",  # Separate file for MNIST
        )
    except Exception as e:
        print(f"\n❌ Failed to authenticate: {e}")
        return

    # Create evaluation interface
    print("\nCreating blackbox evaluation interface...")
    eval_input_batch = create_blackbox_interface(model)
    print("Interface created.\n")

    # Start evaluation session
    print("=" * 60)
    print("Starting Black Box Evaluation Session")
    print("=" * 60)

    try:
        session = BlackBoxSession(eval_input_batch, token, user_id)
        print("\nSession started successfully.")
        print("  The model will now receive batches of MNIST images")
        print("  for classification and calibration evaluation.")
        print("\n  Press Ctrl+C to stop...")

        # Keep the session running
        # The session will automatically process batches as they arrive
        import time

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("   Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error during evaluation session: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Evaluation session completed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
