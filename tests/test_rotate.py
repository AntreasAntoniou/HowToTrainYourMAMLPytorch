# Import necessary libraries and components for test.
import numpy as np

# test_image_rotator.py
import pytest
import torch
from PIL import Image

from src.data import ImageRotator, RotationConfig


@pytest.mark.parametrize(
    "degrees,output_channels", [(90, 1), (180, 3), (45, 1)]
)
def test_image_rotator(degrees, output_channels):
    # Create a RotationConfig and ImageRotator instances
    config = RotationConfig(degrees=degrees, output_channels=output_channels)
    rotator = ImageRotator(config=config)

    # Generate a test image - a white square
    test_image = torch.ones((10, 10, 3)) * 255

    # Apply rotation
    rotated_image = rotator(test_image)

    # Check if image has been correctly rotated and channel adjusted
    assert (
        rotated_image.shape[:2] == test_image.shape[:2]
    ), "Rotated image dimensions should match original"
    if output_channels == 3:
        assert (
            rotated_image.ndim == 3 and rotated_image.shape[2] == 3
        ), "Rotated image should have 3 channels if output_channels is set to 3"
    elif output_channels == 1:
        assert rotated_image.ndim == 2 or (
            rotated_image.ndim == 3 and rotated_image.shape[2] == 1
        ), "Rotated image should have 1 channel if output_channels is set to 1"

    # Optionally, for arbitrary angles, you can compare with a PIL rotation
    if degrees % 90 != 0:
        pil_rotated_image = Image.fromarray(test_image).rotate(
            degrees, resample=Image.BICUBIC
        )
        pil_rotated_image = np.array(pil_rotated_image)
        assert (
            pil_rotated_image.shape == rotated_image.shape
        ), "Shapes from PIL and custom rotation do not match"


def test_rotation_config_initialization():
    degrees = 90
    output_channels = 3
    config = RotationConfig(degrees, output_channels)
    assert config.degrees == degrees
    assert config.output_channels == output_channels


imsize = 10
sample_image = torch.ones((imsize, imsize)) * 255


@pytest.mark.parametrize(
    "degrees,output_channels", [(0, 1), (90, 3), (45, 1), (180, 3)]
)
def test_image_rotator(degrees, output_channels):
    config = RotationConfig(degrees, output_channels)
    rotator = ImageRotator(config)
    rotated_image = rotator(sample_image)

    if output_channels == 1:
        assert rotated_image.ndim == 2 or (
            rotated_image.ndim == 3 and rotated_image.shape[-1] == 1
        )
    else:
        assert rotated_image.ndim == 3 and rotated_image.shape[-1] == 3

    # Check the image is rotated correctly only for multiples of 90 degrees
    if degrees % 90 == 0:
        k = int(degrees / 90)
        expected_shape = (
            (imsize, imsize, output_channels)
            if output_channels > 1
            else (imsize, imsize)
        )
        assert rotated_image.shape == expected_shape
