"""Dataset class for mini-ImageNet dataset."""

import argparse
import os
from typing import Any, List

import torch
from PIL import Image

from dataloaders.utils import (
    get_few_shot_samples_indices_per_class,
    get_session_classes,
)


class MiniImagenetDataset:
    """Dataset class for mini-ImageNet dataset.

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

        setname = "train" if train else "test"
        csv_path = os.path.join(self.root, "split", setname + ".csv")
        lines = [x.strip() for x in open(csv_path, "r").readlines()][1:]  # noqa: SIM115
        self.images = [x.split(",")[0] for x in lines]

        self.labels = [x.split(",")[1] for x in lines]
        all_labels = sorted(set(self.labels))
        self.label_map = {k: v for v, k in enumerate(all_labels)}
        self.labels = [self.label_map[x] for x in self.labels]  # type: ignore

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
                    "mini_imagenet",
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
        img = Image.open(os.path.join(self.root, "images", img_path)).convert("RGB")
        images: List[torch.Tensor] = [
            self.transform(img) for i in range(self.args.num_views)
        ]

        return {"image": images, "label": label}
