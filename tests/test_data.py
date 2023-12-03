# test_dataset_transforms_config.py
# test_transform_scheduler.py


import numpy as np
import PIL.Image as Image
import pytest
import torch
from torch import allclose, tensor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize, to_tensor

from src.data import (
    DatasetTransformsConfig,
    TransformScheduler,
    get_dataloader_dict,
)


def test_dataset_transforms_config_initialization():
    image_size = (84, 84)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    config = DatasetTransformsConfig(image_size, mean, std)
    assert config.image_size == image_size
    assert config.mean == mean
    assert config.std == std


def test_apply_transforms_omniglot():
    configs = DatasetTransformsConfig(
        (28, 28), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
    scheduler = TransformScheduler(configs, "omniglot")

    sample_image = Image.new("L", (28, 28))
    transformed_image = scheduler.apply_transforms(sample_image, train=True)
    assert allclose(transformed_image, to_tensor(sample_image))

    transformed_image = scheduler.apply_transforms(sample_image, train=False)
    assert allclose(transformed_image, to_tensor(sample_image))


def test_apply_transforms_other_datasets():
    configs = DatasetTransformsConfig(
        (84, 84), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    scheduler = TransformScheduler(configs, "mini_imagenet")

    sample_image = Image.new("RGB", (84, 84))
    transformed_image = scheduler.apply_transforms(sample_image, train=True)
    transformed_image_pil = to_tensor(sample_image)
    transformed_image_pil = normalize(
        transformed_image_pil, configs.mean, configs.std
    )
    assert allclose(transformed_image, transformed_image_pil)


@pytest.mark.parametrize(
    "dataset_name", ["omniglot", "mini_imagenet", "cubirds200"]
)
def test_get_dataloader_dict(dataset_name):
    batch_size = 16
    num_workers = 4
    seed = 42
    # This will call your actual get_dataloader_dict function with the mocked datasets
    dataloaders = get_dataloader_dict(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # Optionally, assert the batch size and shuffle properties are being set correctly
    for name, dataloader in dataloaders.items():
        for _, batch in enumerate(dataloader):
            assert len(batch[0]) == batch_size
            break  # we only need to check the first batch
