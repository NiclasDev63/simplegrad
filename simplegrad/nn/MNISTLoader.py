from typing import Literal
from simplegrad import Tensor
import numpy as np
from datasets import load_dataset


def transform(example):
    imgs = np.array(example["image"], dtype=np.float32) / 255.0
    imgs = imgs.reshape(-1, 28 * 28)
    return {"image": imgs, "label": example["label"]}


def get_mnist_split(split: Literal["train", "test"]):
    batch_size = 5_000

    if split == "train":
        train = load_dataset("ylecun/mnist", split="train")
        train = train.map(transform, batched=True, batch_size=batch_size)
        train.set_format(type="numpy", columns=["image", "label"])
        return train
    elif split == "test":
        test = load_dataset("ylecun/mnist", split="test")
        test = test.map(transform, batched=True, batch_size=batch_size)
        test.set_format(type="numpy", columns=["image", "label"])
        return test
    else:
        raise ValueError("Unkown split type")


class MNISTLoader:
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        split: Literal["train", "test"] = "train",
    ):
        """
        Initialize MNISTLoader with data loading options.

        Parameters:
        -----------
        batch_size : int
            Number of samples per batch
        shuffle : bool, default=False
            Whether to shuffle the data at the beginning of each epoch
        drop_last : bool, default=False
            Whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Load data
        self.dataset = get_mnist_split(split=split)
        self.num_samples = len(self.dataset)

        # dataset info
        self.images_size_flattened = 28 * 28
        self.num_classes = 10

        # Initialize indices for iteration
        self.indices = np.arange(len(self.dataset))

        # Current position for iteration
        self.current_train_pos = 0
        self.current_test_pos = 0

        if self.shuffle:
            self._shuffle_train_data()

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return int(np.ceil(n / self.batch_size))

    def _shuffle_train_data(self):
        """Shuffle the training data indices."""
        np.random.shuffle(self.indices)
        self.current_train_pos = 0

    def _get_batch(self):
        end_idx = min(self.current_train_pos + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_train_pos : end_idx]

        batch_images = Tensor(self.dataset["image"][batch_indices].squeeze(-1))

        labels = self.dataset["label"][batch_indices]
        labels_onehot = np.eye(self.num_classes, dtype=np.float32)[labels]
        batch_labels = Tensor(labels_onehot)

        return batch_images, batch_labels, end_idx

    def __iter__(self):
        """Make the loader iterable."""
        return self

    def __next__(self):
        """Get the next batch of training data."""
        if self.current_train_pos >= self.num_samples:
            # End of epoch, reset and shuffle if needed
            self.current_train_pos = 0
            if self.shuffle:
                self._shuffle_train_data()
            raise StopIteration

        batch_images, batch_labels, self.current_train_pos = self._get_batch()

        return batch_images, batch_labels
