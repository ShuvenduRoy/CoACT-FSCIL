"""Dataset class for CUB-200 dataset."""

import argparse
import os
from typing import Any, List

import torch
from PIL import Image

from dataloaders.utils import (
    get_few_shot_samples_indices_per_class,
    get_session_classes,
)


class Cub200Dataset:
    """Dataset class for CUB-200 dataset.

    Parameters
    ----------
    root : str
        Root directory of dataset where directory ``cifar-10-batches-py`` exists or
        will be saved to if download is set to True.
    train : bool, optional
        If True, creates dataset from training set, otherwise creates from test set.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    download : bool, optional
        If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    session : int, optional
        Current session.
    transformations : optional
        transformations.
    args : argparse.ArgumentParser, optional
        Arguments passed to the trainer.

    Returns
    -------
    None
    """

    def __init__(
        self,
        root: str,
        args: argparse.Namespace,
        train: bool = True,
        download: bool = False,
        session: int = 0,
        transformations: Any = None,
    ):
        self.transform = transformations
        self.root = root
        self.args = args

        image_file = os.path.join(root, "CUB_200_2011/images.txt")
        split_file = os.path.join(root, "CUB_200_2011/train_test_split.txt")
        class_file = os.path.join(root, "CUB_200_2011/image_class_labels.txt")

        self.images = ["CUB_200_2011/images/" + x for x in self.text_read(image_file)]
        self.split = [int(x) for x in self.text_read(split_file)]
        self.labels = [int(x) - 1 for x in self.text_read(class_file)]
        train_set = [i for i, x in enumerate(self.split) if x == 1]
        test_set = [i for i, x in enumerate(self.split) if x == 0]
        if train:
            self.images = [self.images[i] for i in train_set]
            self.labels = [self.labels[i] for i in train_set]
        else:
            self.images = [self.images[i] for i in test_set]
            self.labels = [self.labels[i] for i in test_set]

        classes_at_current_session = get_session_classes(args, session)

        if train:  # few-shot training samples
            if (
                args.fsl_setup == "FSCIL"
                and session == 0
                and args.reduced_base_shot < 0
            ):
                sample_ids = [
                    i
                    for i, label in enumerate(self.labels)
                    if label in classes_at_current_session
                ]
            else:
                shot = args.shot
                if (
                    args.fsl_setup == "FSCIL"
                    and session == 0
                    and args.reduced_base_shot > 0
                ):
                    shot = args.reduced_base_shot

                sample_ids = get_few_shot_samples_indices_per_class(
                    "cub200",
                    classes_at_current_session,
                    shot,
                )
        else:  # validation; all samples of the classes till curr session
            sample_ids = [
                i
                for i, label in enumerate(self.labels)
                if label <= max(classes_at_current_session)
            ]

        self.images = [self.images[i] for i in sample_ids]
        self.labels = [self.labels[i] for i in sample_ids]

    def text_read(self, file: str) -> List[str]:
        """Read text file."""
        with open(file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line_ = line.strip("\n")
                lines[i] = line_.split(" ")[1]
        return lines

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Any:
        """Load one sample.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        img_path, label = self.images[index], self.labels[index]
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        images: List[torch.Tensor] = [
            self.transform(img) for i in range(self.args.num_views)
        ]

        return {"image": images, "label": label}
