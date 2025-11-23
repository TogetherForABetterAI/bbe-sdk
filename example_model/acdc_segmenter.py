"""
ACDC Cardiac Segmentation Model

This module provides a simplified segmentation model for ACDC (Automated Cardiac
Diagnosis Challenge) dataset. It uses a U-Net architecture with an EfficientNet backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for cardiac segmentation.

    Input: (batch, 1, H, W) - grayscale cardiac MRI
    Output: (batch, num_classes, H, W) - segmentation logits
    """

    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        out = self.out(dec1)
        return out


class ACDCSegmenter:
    """
    ACDC Segmentation model wrapper compatible with BlackBoxSession.

    Provides a predict() method that takes batched images and returns
    probability distributions for each pixel classification.
    """

    def __init__(self, model_path=None, num_classes=4):
        """
        Args:
            model_path: Path to saved model weights (optional)
            num_classes: Number of segmentation classes (default: 4)
                - 0: Background
                - 1: Right Ventricle (RV)
                - 2: Myocardium
                - 3: Left Ventricle (LV)
        """
        self.num_classes = num_classes
        self.model = SimpleUNet(in_channels=1, num_classes=num_classes)

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                print(f"Loaded model weights from {model_path}")
            except FileNotFoundError:
                print(f"Model weights not found at {model_path}")
                print("   Using randomly initialized weights (for testing only)")
        else:
            print("No model path provided. Using randomly initialized weights.")
            print("   This is suitable for testing the data pipeline only.")

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, images):
        """
        Predict segmentation probabilities for a batch of images.

        Args:
            images: numpy array of shape (batch, C, H, W) or (batch, H, W, C)
                    Expected to be grayscale cardiac MRI images

        Returns:
            List of probability arrays, one per image. Each array has shape
            (num_pixels, num_classes) representing class probabilities for each pixel.
        """
        # Convert numpy to tensor
        if isinstance(images, np.ndarray):
            # Handle different input formats
            if images.ndim == 3:
                # (batch, H, W) -> add channel dim
                images = np.expand_dims(images, axis=1)
            elif images.ndim == 4:
                # Check if it's (batch, H, W, C) and needs transpose
                if images.shape[-1] == 1 and images.shape[1] != 1:
                    # (batch, H, W, 1) -> (batch, 1, H, W)
                    images = np.transpose(images, (0, 3, 1, 2))

            img_tensor = torch.from_numpy(images).float()
        else:
            img_tensor = images

        # Move to device
        img_tensor = img_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(img_tensor)  # (batch, num_classes, H, W)
            probs = F.softmax(logits, dim=1)  # (batch, num_classes, H, W)

        # Convert to list of per-image probability distributions
        batch_size = probs.shape[0]
        predictions = []

        for i in range(batch_size):
            # Get probabilities for this image: (num_classes, H, W)
            img_probs = probs[i].cpu().numpy()

            # Reshape to (num_pixels, num_classes) for compatibility with SDK
            # Flatten spatial dimensions: (num_classes, H, W) -> (H*W, num_classes)
            h, w = img_probs.shape[1], img_probs.shape[2]
            img_probs = img_probs.reshape(self.num_classes, -1).T  # (H*W, num_classes)

            # For compatibility with classification interface,
            # we'll compute the dominant class distribution
            # This averages probabilities across all pixels
            avg_probs = img_probs.mean(axis=0).tolist()  # (num_classes,)

            predictions.append(avg_probs)

        return predictions

    def predict_full_mask(self, images):
        """
        Predict full segmentation masks (for visualization/debugging).

        Args:
            images: numpy array of shape (batch, C, H, W) or (batch, H, W, C)

        Returns:
            numpy array of shape (batch, H, W) with class predictions
        """
        # Convert and prepare images
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = np.expand_dims(images, axis=1)
            elif images.ndim == 4 and images.shape[-1] == 1:
                images = np.transpose(images, (0, 3, 1, 2))
            img_tensor = torch.from_numpy(images).float()
        else:
            img_tensor = images

        img_tensor = img_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            preds = logits.argmax(dim=1)  # (batch, H, W)

        return preds.cpu().numpy()


# For backward compatibility
def load_model(model_path):
    """Load a pretrained ACDC segmentation model."""
    return ACDCSegmenter(model_path=model_path)


def build_model(num_classes=4):
    """Build a new ACDC segmentation model."""
    return ACDCSegmenter(num_classes=num_classes)
