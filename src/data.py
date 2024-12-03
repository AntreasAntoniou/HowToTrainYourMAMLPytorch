import json
import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from rich import print
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class RotationConfig:
    def __init__(self, degrees: float, output_channels: int):
        """
        Configuration object for image rotation.

        :param degrees: The degree of rotation.
        :param output_channels: Number of channels desired in the output image.
        """
        self.degrees = degrees
        self.output_channels = output_channels


class ImageRotator:
    def __init__(self, config: RotationConfig):
        """
        Rotates images by a given number of degrees.

        :param config: The RotationConfig object containing rotation parameters.
        """
        self.config = config

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate the image according to the provided configuration.

        :param image: The input image as a numpy array.
        :return: The rotated image as a numpy array.
        """
        if self.config.degrees % 90 != 0:
            # Arbitrary rotations
            rotate = transforms.functional.rotate
            print(image)
            image_pil = (
                Image.fromarray(image)
                if isinstance(image, np.ndarray)
                else Image.fromarray(image.numpy())
                if isinstance(image, torch.Tensor)
                else image
            )
            image_rotated_pil = rotate(image_pil, self.config.degrees)
            image_rotated = np.array(image_rotated_pil)
        else:
            # 90 degree rotations can use np.rot90 for better performance
            image = image if isinstance(image, np.ndarray) else np.array(image)
            k = int(self.config.degrees / 90)
            image_rotated = np.rot90(image, k=k)

        # Ensure the output has the correct number of channels
        if image_rotated.ndim == 2 and self.config.output_channels == 3:
            image_rotated = np.stack((image_rotated,) * 3, axis=-1)
        elif image_rotated.ndim == 3 and self.config.output_channels == 1:
            image_rotated = np.expand_dims(
                np.mean(image_rotated, axis=-1), axis=-1
            )

        return image_rotated


class DatasetTransformsConfig:
    def __init__(
        self,
        image_size: Tuple[int, int],
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std


def remap_to_few_shot_classes(
    support_set_labels: torch.Tensor, target_set_labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find the unique labels in the combined set and, implicitly, the new mapping
    support_set_shape = support_set_labels.shape
    target_set_shape = target_set_labels.shape

    unique_labels, inverse_indices = torch.unique(
        torch.cat([support_set_labels.view(-1), target_set_labels.view(-1)]),
        sorted=True,
        return_inverse=True,
    )

    # Map labels to their new values in a vectorized manner
    inverse_indices = inverse_indices.to(
        support_set_labels.dtype
    )  # Ensure same dtype as input labels

    remapped_support_set_labels = inverse_indices[
        : support_set_labels.view(-1).size(0)
    ].view(support_set_shape)

    remapped_target_set_labels = inverse_indices[
        support_set_labels.view(-1).size(0) :
    ].view(target_set_shape)

    return remapped_support_set_labels, remapped_target_set_labels


class TransformScheduler:
    def __init__(self, config: DatasetTransformsConfig, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.transform_train, self.transform_evaluate = self.get_transforms()

    def get_transforms(
        self,
    ) -> Tuple[List[transforms.Compose], List[transforms.Compose]]:
        if "omniglot" in self.dataset_name:
            transform_train = [
                transforms.Resize(self.config.image_size),
                ImageRotator(RotationConfig(0, 1)),
                transforms.ToTensor(),
            ]
            transform_evaluate = [
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
            ]
        else:
            transform_train = [
                transforms.Compose(
                    [
                        transforms.Resize(self.config.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            self.config.mean, self.config.std
                        ),
                    ]
                )
            ]
            transform_evaluate = [
                transforms.Compose(
                    [
                        transforms.Resize(self.config.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            self.config.mean, self.config.std
                        ),
                    ]
                )
            ]

        return transform_train, transform_evaluate

    def apply_transforms(self, image: np.ndarray, train: bool) -> np.ndarray:
        transforms_to_apply = (
            self.transform_train if train else self.transform_evaluate
        )
        for transform in transforms_to_apply:
            image = transform(image)
        return image


class DatasetPathConfig:
    def __init__(self, dataset_name: str, dataset_dir: str):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir


@dataclass
class FewShotTaskConfig:
    num_samples_per_class: int
    num_classes_per_set: int
    num_target_samples: int


class FewShotLearningDatasetParallel(Dataset):
    def __init__(
        self,
        hf_dataset,
        num_episodes: int,
        transforms: TransformScheduler,
        few_shot_task_config: FewShotTaskConfig,
        seed: int,
        use_train_transforms: bool = True,
        data_dir: str = None,
    ):
        self.hf_dataset = hf_dataset
        self.transforms = transforms
        self.support_set_transform = (
            lambda x: self.transforms.apply_transforms(
                x, train=use_train_transforms
            )
        )
        self.target_set_transform = lambda x: self.transforms.apply_transforms(
            x, train=False
        )
        self.few_shot_task_config = few_shot_task_config
        if data_dir is None:
            # set to tmpdir if not specified
            data_dir = Path(tempfile.gettempdir())

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.data_dir = data_dir

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

        dataset_label_to_idx_json_path = (
            self.data_dir / "dataset_label_to_idx_dict.json"
        )

        if dataset_label_to_idx_json_path.exists():
            self.dataset_label_to_idx_dict = json.load(
                open(dataset_label_to_idx_json_path)
            )
        else:
            self.dataset_label_to_idx_dict = self.dataset_label_to_idx()
            with open(dataset_label_to_idx_json_path, "w") as f:
                json.dump(self.dataset_label_to_idx_dict, f)

        self.seed = seed
        self.num_episodes = num_episodes

    def dataset_label_to_idx(self):
        dataset_label_to_idx = {}
        for idx, sample in tqdm(enumerate(self.hf_dataset)):
            label = sample["label"]
            if label not in dataset_label_to_idx:
                dataset_label_to_idx[label] = [idx]
                continue
            dataset_label_to_idx[label].append(idx)

        return dataset_label_to_idx

    def _get_task_set(self, index):
        """
        Generates a task-set to be used for training or evaluation.
        """
        # Assuming the dataset has attributes like num_samples_per_class and num_classes_per_set
        rng = np.random.RandomState(seed=self.seed + index)

        # Randomly select classes to include in the task

        classes = rng.choice(
            list(self.dataset_label_to_idx_dict.keys()),
            size=self.few_shot_task_config.num_classes_per_set,
            replace=False,
        )

        support_set_images, support_set_labels = [], []
        target_set_images, target_set_labels = [], []

        # For each class, randomly select samples for support set and target set
        for class_id, class_name in enumerate(classes):
            # Select image indices for support and target set
            image_indices = rng.choice(
                self.dataset_label_to_idx_dict[class_name],
                size=self.few_shot_task_config.num_samples_per_class
                + self.few_shot_task_config.num_target_samples,
                replace=False,
            )
            shuffle(image_indices)
            support_indices = image_indices[
                : self.few_shot_task_config.num_samples_per_class
            ]
            target_indices = image_indices[
                self.few_shot_task_config.num_samples_per_class :
            ]
            support_set_class_images = []
            support_set_class_labels = []
            target_set_class_images = []
            target_set_class_labels = []
            # Load and preprocess images for support and target set
            for idx in support_indices:
                idx = int(idx)
                image = self.hf_dataset[idx]["image"]
                image = self.support_set_transform(image)
                label = self.hf_dataset[idx]["label"]
                support_set_class_images.append(image)
                support_set_class_labels.append(label)

            for idx in target_indices:
                idx = int(idx)
                image = self.hf_dataset[idx]["image"]
                image = self.target_set_transform(image)
                label = self.hf_dataset[idx]["label"]
                target_set_class_images.append(image)
                target_set_class_labels.append(label)

            support_set_class_labels = torch.tensor(support_set_class_labels)
            target_set_class_labels = torch.tensor(target_set_class_labels)

            support_set_images.append(torch.stack(support_set_class_images))
            support_set_labels.append(support_set_class_labels)
            target_set_images.append(torch.stack(target_set_class_images))
            target_set_labels.append(target_set_class_labels)

        # Stack images and labels to construct task set tensors
        support_set_images = torch.stack(support_set_images)
        support_set_labels = torch.stack(support_set_labels)

        target_set_images = torch.stack(target_set_images)
        target_set_labels = torch.stack(target_set_labels)

        (
            support_set_labels,
            target_set_labels,
        ) = remap_to_few_shot_classes(support_set_labels, target_set_labels)

        return (
            support_set_images,
            target_set_images,
            support_set_labels,
            target_set_labels,
        )

    def __getitem__(self, index):
        return self._get_task_set(index)

    def __len__(self):
        return self.num_episodes


def get_dataloader_dict(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    num_train_episodes: int = 100000,
    num_eval_episodes: int = 600,
    keep_in_memory: bool = True,
    data_cache_dir: str = None,
    num_samples_per_class: int = 1,
    num_classes_per_set: int = 5,
    num_target_samples: int = 1,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for the specified few-shot learning dataset with dataset-specific image sizes and splits for Omniglot.

    :param dataset_name: Name of the dataset ('omniglot', 'mini_imagenet', or 'cubirds200').
    :param batch_size: Size of each batch per iteration.
    :param num_workers: Number of workers for data loading.
    :param seed: Random seed for reproducibility.
    :return: A dictionary with keys 'train', 'val', and 'test' mapping to the corresponding DataLoaders.
    """
    # Load dataset
    import datasets

    if data_cache_dir is None:
        # set to tmpdir if not specified
        data_cache_dir = Path(tempfile.gettempdir())

    if isinstance(data_cache_dir, str):
        data_cache_dir = Path(data_cache_dir)

    if dataset_name == "omniglot":
        hf_dataset = datasets.load_dataset(
            f"GATE-engine/{dataset_name}",
            split="full",
            keep_in_memory=keep_in_memory,
            cache_dir=data_cache_dir,
            num_proc=mp.cpu_count(),
        )

        # Define the splits for Omniglot according to the specified indices
        train_indices = list(range(0, 1200 * 20))
        val_indices = list(range(1200 * 20, 1400 * 20))
        test_indices = list(range(1400 * 20, 1622 * 20))

        datasets_dict = {
            "train": (
                datasets.Dataset.from_dict(hf_dataset[train_indices]),
                num_train_episodes,
            ),
            "val": (
                datasets.Dataset.from_dict(hf_dataset[val_indices]),
                num_eval_episodes,
            ),
            "test": (
                datasets.Dataset.from_dict(hf_dataset[test_indices]),
                num_eval_episodes,
            ),
        }
    else:
        hf_datasets = datasets.load_dataset(
            f"GATE-engine/{dataset_name}",
            keep_in_memory=keep_in_memory,
            cache_dir=data_cache_dir,
            num_proc=mp.cpu_count(),
        )
        datasets_dict = {
            split: (
                hf_datasets[split],
                num_train_episodes if split == "train" else num_eval_episodes,
            )
            for split in ["train", "validation", "test"]
        }
        datasets_dict["val"] = datasets_dict["validation"]

    # Set the image size based on the dataset
    image_size = (28, 28) if dataset_name == "omniglot" else (84, 84)

    # Create a generic transforms config for all datasets
    config_transforms = DatasetTransformsConfig(
        image_size=image_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    few_shot_config = FewShotTaskConfig(
        num_samples_per_class=num_samples_per_class,
        num_classes_per_set=num_classes_per_set,
        num_target_samples=num_target_samples,
    )

    # Helper function to create dataset object for each split with Omniglot specific logic
    def create_dataset(dataset, num_episodes, split):
        return FewShotLearningDatasetParallel(
            dataset,
            num_episodes=num_episodes,
            transforms=TransformScheduler(
                dataset_name=dataset_name,
                config=config_transforms,
            ),
            few_shot_task_config=few_shot_config,
            seed=seed,
            use_train_transforms=(split == "train"),
            data_dir=data_cache_dir / dataset_name / split,
        )

    # Create Dataset objects for each split
    datasets_objects = {
        split: create_dataset(dataset, num_episodes, split)
        for split, (dataset, num_episodes) in datasets_dict.items()
    }

    # Create DataLoaders from datasets
    dataloaders_dict = {
        split: DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=True,
        )
        for split, dataset_obj in datasets_objects.items()
    }

    return dataloaders_dict
