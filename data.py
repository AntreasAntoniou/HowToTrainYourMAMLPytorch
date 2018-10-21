import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
from torchvision import transforms


class rotate_image(object):

    def __init__(self, k, channels):
        self.k = k
        self.channels = channels

    def __call__(self, image):
        if self.channels == 1 and len(image.shape) == 3:
            image = image[:, :, 0]
            image = np.expand_dims(image, axis=2)

        elif self.channels == 1 and len(image.shape) == 4:
            image = image[:, :, :, 0]
            image = np.expand_dims(image, axis=3)

        image = np.rot90(image, k=self.k).copy()
        return image


class torch_rotate_image(object):

    def __init__(self, k, channels):
        self.k = k
        self.channels = channels

    def __call__(self, image):
        rotate = transforms.RandomRotation(degrees=self.k)
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        image = Image.fromarray(image)
        image = rotate(image)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        return image


def augment_image(image, k, channels, augment_bool, args, dataset_name):
    transform_train, transform_evaluation = get_transforms_for_dataset(dataset_name=dataset_name,
                                                                       args=args, k=k)
    if len(image.shape) > 3:
        images = [item for item in image]
        output_images = []
        for image in images:
            if augment_bool is True:
                # print("augment")
                # meanstd transformation
                for transform_current in transform_train:
                    image = transform_current(image)
            else:
                for transform_current in transform_evaluation:
                    image = transform_current(image)
            output_images.append(image)
        image = torch.stack(output_images)
    else:
        if augment_bool is True:
            # meanstd transformation
            for transform_current in transform_train:
                image = transform_current(image)
        else:
            for transform_current in transform_evaluation:
                image = transform_current(image)
    return image


def get_transforms_for_dataset(dataset_name, args, k):
    if "cifar10" in dataset_name or "cifar100" in dataset_name:
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(args.classification_mean, args.classification_std)]

        transform_evaluate = [
            transforms.ToTensor(),
            transforms.Normalize(args.classification_mean, args.classification_std)]

    elif 'omniglot' in dataset_name:

        transform_train = [rotate_image(k=k, channels=args.image_channels), transforms.ToTensor()]
        transform_evaluate = [transforms.ToTensor()]

    elif 'imagenet' in dataset_name:
        transform_train = [transforms.Compose([
            transforms.ToTensor()])]

        transform_evaluate = [transforms.Compose([
            transforms.ToTensor()])]

    return transform_train, transform_evaluate



class FewShotLearningDatasetParallel(Dataset):
    def __init__(self, args):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.args = args
        self.indexes_of_folders_indicating_class = args.indexes_of_folders_indicating_class
        self.labels_as_int = args.labels_as_int
        self.train_val_test_split = args.train_val_test_split
        self.current_set_name = "train"
        self.num_target_samples = args.num_target_samples
        self.reset_stored_filepaths = args.reset_stored_filepaths
        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.rng = np.random.RandomState(seed=self.seed['val'])
        self.datasets = self.load_dataset()
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.reverse_channels = args.reverse_channels
        self.image_height, self.image_width, self.image_channel = args.image_height, args.image_width, args.image_channels
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.augment_images = False
        self.num_samples_per_class = args.num_samples_per_class
        self.num_classes_per_set = args.num_classes_per_set

        self.indexes = {"train": 0, "val": 0, 'test': 0}
        self.dataset_size_dict = {"train": {key: len(self.datasets['train'][key]) for key in list(self.datasets['train'].keys())},
                                  "val": {key: len(self.datasets['val'][key]) for key in list(self.datasets['val'].keys())},
                                  'test': {key: len(self.datasets['test'][key]) for key in list(self.datasets['test'].keys())}}
        self.label_set = self.get_label_set()
        self.data_length = {name: np.sum([len(self.datasets[name][key])
                                          for key in self.datasets[name]]) for name in self.datasets.keys()}

        print("data", self.data_length)
        self.observed_seed_set = None

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """
        rng = np.random.RandomState(seed=self.seed['val'])
        data_image_paths, index_to_label_name_dict_file, label_to_index = self.load_datapaths()
        dataset_splits = dict()
        if self.args.sets_are_pre_split == True:
            for key, value in data_image_paths.items():
                key = self.get_label_from_index(index=key)
                bits = key.split("/")
                set_name = bits[0]
                class_label = bits[1]
                if set_name not in dataset_splits:
                    dataset_splits[set_name] = {class_label: value}
                else:
                    dataset_splits[set_name][class_label] = value
            return dataset_splits
        else:
            total_label_types = len(data_image_paths)
            num_classes_idx = np.arange(len(data_image_paths.keys()), dtype=np.int32)
            rng.shuffle(num_classes_idx)
            keys = list(data_image_paths.keys())
            values = list(data_image_paths.values())
            new_keys = [keys[idx] for idx in num_classes_idx]
            new_values = [values[idx] for idx in num_classes_idx]
            data_image_paths = dict(zip(new_keys, new_values))
            # data_image_paths = self.shuffle(data_image_paths)
            x_train_id, x_val_id, x_test_id = int(self.train_val_test_split[0] * total_label_types), \
                                              int(np.sum(self.train_val_test_split[:2]) * total_label_types), \
                                              int(total_label_types)
            print(x_train_id, x_val_id, x_test_id)
            x_train_classes = (class_key for class_key in list(data_image_paths.keys())[:x_train_id])
            x_val_classes = (class_key for class_key in list(data_image_paths.keys())[x_train_id:x_val_id])
            x_test_classes = (class_key for class_key in list(data_image_paths.keys())[x_val_id:x_test_id])
            x_train, x_val, x_test = {class_key: data_image_paths[class_key] for class_key in x_train_classes}, \
                                     {class_key: data_image_paths[class_key] for class_key in x_val_classes}, \
                                     {class_key: data_image_paths[class_key] for class_key in x_test_classes},
            dataset_splits = {"train": x_train, "val": x_val, "test": x_test}
            return dataset_splits

    def load_datapaths(self):
        """
        If saved json dictionaries of the data are available, then this method loads the dictionaries such that the
        data is ready to be read. If the json dictionaries do not exist, then this method calls get_data_paths()
        which will build the json dictionary containing the class to filepath samples, and then store them.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        dataset_dir = os.environ['DATASET_DIR']
        data_path_file = "{}/{}.json".format(dataset_dir, self.dataset_name)
        self.index_to_label_name_dict_file = "{}/map_to_label_name_{}.json".format(dataset_dir, self.dataset_name)
        self.label_name_to_map_dict_file = "{}/label_name_to_map_{}.json".format(dataset_dir, self.dataset_name)

        if not os.path.exists(data_path_file):
            self.reset_stored_filepaths = True

        if self.reset_stored_filepaths == True:
            if os.path.exists(data_path_file):
                os.remove(data_path_file)
            self.reset_stored_filepaths = False

        try:
            data_image_paths = self.load_from_json(filename=data_path_file)
            label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
            index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
            return data_image_paths, index_to_label_name_dict_file, label_to_index
        except:
            print("Mapped data paths can't be found, remapping paths..")
            data_image_paths, code_to_label_name, label_name_to_code = self.get_data_paths()
            self.save_to_json(dict_to_store=data_image_paths, filename=data_path_file)
            self.save_to_json(dict_to_store=code_to_label_name, filename=self.index_to_label_name_dict_file)
            self.save_to_json(dict_to_store=label_name_to_code, filename=self.label_name_to_map_dict_file)
            return self.load_datapaths()

    def save_to_json(self, filename, dict_to_store):
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(dict_to_store, fp=f)

    def load_from_json(self, filename):
        with open(filename, mode="r") as f:
            load_dict = json.load(fp=f)

        return load_dict

    def load_test_image(self, filepath):
        """
        Tests whether a target filepath contains an uncorrupted image. If image is corrupted, attempt to fix.
        :param filepath: Filepath of image to be tested
        :return: Return filepath of image if image exists and is uncorrupted (or attempt to fix has succeeded),
        else return None
        """
        image = None
        try:
            image = Image.open(filepath)
        except RuntimeWarning:
            os.system("convert {} -strip {}".format(filepath, filepath))
            print("converting")
            image = Image.open(filepath)
        except:
            print("Broken image")

        if image is not None:
            return filepath
        else:
            return None

    def get_data_paths(self):
        """
        Method that scans the dataset directory and generates class to image-filepath list dictionaries.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        print("Get images from", self.data_path)
        data_image_path_list_raw = []
        labels = set()
        for subdir, dir, files in os.walk(self.data_path):
            for file in files:
                if (".jpeg") in file.lower() or (".png") in file.lower() or (".jpg") in file.lower():
                    filepath = os.path.join(subdir, file)
                    label = self.get_label_from_path(filepath)
                    data_image_path_list_raw.append(filepath)
                    labels.add(label)

        labels = sorted(labels)
        idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
        label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
        data_image_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
        with tqdm.tqdm(total=len(data_image_path_list_raw)) as pbar_error:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for image_file in executor.map(self.load_test_image, (data_image_path_list_raw)):
                    pbar_error.update(1)
                    if image_file is not None:
                        label = self.get_label_from_path(image_file)
                        data_image_path_dict[label_name_to_idx[label]].append(image_file)

        return data_image_path_dict, idx_to_label_name, label_name_to_idx

    def get_label_set(self):
        """
        Generates a set containing all class numerical indexes
        :return: A set containing all class numerical indexes
        """
        index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return set(list(index_to_label_name_dict_file.keys()))

    def get_index_from_label(self, label):
        """
        Given a class's (human understandable) string, returns the numerical index of that class
        :param label: A string of a human understandable class contained in the dataset
        :return: An int containing the numerical index of the given class-string
        """
        label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
        return label_to_index[label]

    def get_label_from_index(self, index):
        index_to_label_name = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return index_to_label_name[index]

    def get_label_from_path(self, filepath):
        label_bits = filepath.split("/")
        label = "/".join([label_bits[idx] for idx in self.indexes_of_folders_indicating_class])
        if self.labels_as_int:
            label = int(label)
        return label

    def load_image(self, image_path, channels):
        image = Image.open(image_path)
        if 'omniglot' in self.dataset_name:
            image = image.resize((self.image_height, self.image_width), resample=Image.LANCZOS)
            image = np.array(image, np.float32)
        else:
            image = image.resize((self.image_height, self.image_width)).convert('RGB')
            image = np.array(image, np.float32)
            image = image / 255.0

        if channels == 1:
            image = np.expand_dims(image, axis=2)

        return image

    def load_batch(self, batch_image_paths):

        image_batch = []

        for image_path in batch_image_paths:
            image = self.load_image(image_path=image_path, channels=self.image_channel)
            image_batch.append(image)

        image_batch = np.array(image_batch, dtype=np.float32)
        image_batch = self.preprocess_data(image_batch)

        return image_batch

    def preprocess_data(self, x):
        """
        Preprocesses data such that their values lie in the -1.0 to 1.0 range so that the tanh activation gen output
        can work properly
        :param x: A data batch to preprocess
        :return: A preprocessed data batch
        """
        # x = x / 255.0
        # x = 2 * x - 1
        x_shape = x.shape
        x = np.reshape(x, (-1, x_shape[-3], x_shape[-2], x_shape[-1]))
        if self.reverse_channels is True:
            reverse_photos = np.ones(shape=x.shape)
            for channel in range(x.shape[-1]):
                reverse_photos[:, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, channel]
            x = reverse_photos
        x = x.reshape(x_shape)
        # print(x.mean(), x.min(), x.max())
        return x

    def reconstruct_original(self, x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        x = (x + 1) / 2
        x = x * 255.0
        return x

    def shuffle(self, x):
        """
        Shuffles the data batch along it's first axis
        :param x: A data batch
        :return: A shuffled data batch
        """
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        return x

    def get_set(self, dataset_name, seed, augment_images=False):
        """
        Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc
        :return: A data batch
        """
        rng = np.random.RandomState(seed)

        selected_classes = rng.choice(list(self.dataset_size_dict[dataset_name].keys()),
                                      size=self.num_classes_per_set, replace=False)
        rng.shuffle(selected_classes)
        k_list = rng.randint(0, 3, size=self.num_classes_per_set)
        k_dict = {selected_class: k_item for (selected_class, k_item) in zip(selected_classes, k_list)}
        episode_labels = [i for i in range(self.num_classes_per_set)]
        class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                  zip(selected_classes, episode_labels)}

        x_images = []
        y_labels = []

        for class_entry in selected_classes:
            choose_samples_list = rng.choice(self.dataset_size_dict[dataset_name][class_entry],
                                             size=self.num_samples_per_class + self.num_target_samples, replace=False)
            class_image_samples = []
            class_labels = []
            for sample in choose_samples_list:
                choose_samples = self.datasets[dataset_name][class_entry][sample]
                x_class_data = self.load_batch([choose_samples])[0]
                k = k_dict[class_entry]
                x_class_data = augment_image(image=x_class_data, k=k * 90,
                                             channels=self.image_channel, augment_bool=augment_images,
                                             dataset_name=self.dataset_name, args=self.args)
                class_image_samples.append(x_class_data)
                class_labels.append(int(class_to_episode_label[class_entry]))
            class_image_samples = torch.stack(class_image_samples)
            x_images.append(class_image_samples)
            y_labels.append(class_labels)

        x_images = torch.stack(x_images)
        y_labels = np.array(y_labels, dtype=np.int32)

        target_set_images = x_images[:, self.num_samples_per_class:]
        target_set_labels = y_labels[:, self.num_samples_per_class:]
        support_set_images = x_images[:, :self.num_samples_per_class]
        support_set_labels = y_labels[:, :self.num_samples_per_class]

        return support_set_images, target_set_images, support_set_labels, target_set_labels

    def __len__(self):
        total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_augmentation(self, augment_images):
        self.augment_images = augment_images

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(dataset_name=set_name, seed=self.init_seed[set_name] + current_iter)

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):
        support_set_images, target_set_image, support_set_labels, target_set_label = \
            self.get_set(self.current_set_name, seed=self.seed[self.current_set_name] + idx,
                         augment_images=self.augment_images)
        data_point = {"support_set_images": support_set_images, "target_set_image": target_set_image,
                      "support_set_labels": support_set_labels, "target_set_label": target_set_label}
        return data_point

    def reset_seed(self):
        self.seed = self.init_seed


class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0):
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = FewShotLearningDatasetParallel(args=args)
        self.batches_per_iter = args.samples_per_iter
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
                          shuffle=False, num_workers=self.num_workers, drop_last=True)

    def continue_from_iter(self, current_iter):
        self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))

    def get_train_batches(self, total_batches=-1, augment_images=False):

        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.dataset.set_augmentation(augment_images=augment_images)
        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            preprocess_sample = self.sample_iter_data(sample=sample_batched, num_gpus=self.dataset.num_of_gpus,
                                                      samples_per_iter=self.batches_per_iter,
                                                      batch_size=self.dataset.batch_size)
            yield preprocess_sample

    def get_val_batches(self, total_batches=-1, augment_images=False):
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")
        self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            preprocess_sample = self.sample_iter_data(sample=sample_batched, num_gpus=self.dataset.num_of_gpus,
                                                      samples_per_iter=self.batches_per_iter,
                                                      batch_size=self.dataset.batch_size)
            yield preprocess_sample

    def get_test_batches(self, total_batches=-1, augment_images=False):
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name='test')
        self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            preprocess_sample = self.sample_iter_data(sample=sample_batched, num_gpus=self.dataset.num_of_gpus,
                                                      samples_per_iter=self.batches_per_iter,
                                                      batch_size=self.dataset.batch_size)
            yield preprocess_sample

    def sample_iter_data(self, sample, num_gpus, batch_size, samples_per_iter):
        output_sample = []
        for key in sample.keys():
            sample[key] = np.array(sample[key].numpy(), dtype=np.float32)
            new_shape = []
            curr_id = 1

            for i in range(len(sample[key].shape) + 2):
                if i == 0:
                    new_shape.append(samples_per_iter)
                elif i == 1:
                    new_shape.append(num_gpus)
                elif i == 2:
                    new_shape.append(batch_size)
                else:
                    new_shape.append(sample[key].shape[curr_id])
                    curr_id += 1

            output_sample.append(np.reshape(sample[key], newshape=new_shape))

        return output_sample
