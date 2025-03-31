"""Helper functions for the methods."""

import argparse
from typing import Any, Tuple

import torch
from torchvision import transforms
from transformers import ViTImageProcessor

from dataloaders.datasets.cub200 import Cub200Dataset
from dataloaders.datasets.hf_dataset import hf_dataset
from dataloaders.datasets.miniimagenet import MiniImagenetDataset


dataset_class_map = {
    "cub200": Cub200Dataset,
    "mini_imagenet": MiniImagenetDataset,
}


def get_transform(args: argparse.Namespace) -> Tuple[Any, Any]:
    """Return the transforms for the dataset.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Arguments passed to the trainer.

    Returns
    -------
    Tuple[Any, Any]
        The crop transform and the secondary transform for the dataset.
    """
    try:
        processor = ViTImageProcessor.from_pretrained(args.hf_model_checkpoint)
        normalize = transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        )
    except Exception as e:
        print(f"Error with ViTImageProcessor: {e}")
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25],
        )

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.size_crops[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ],
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.size_crops[0]),
            transforms.CenterCrop(args.size_crops[0]),
            transforms.ToTensor(),
            normalize,
        ],
    )

    return train_transforms, val_transforms


def get_dataloader(args: argparse.Namespace, session: int = 0) -> Tuple[Any, Any, Any]:
    """Get the base dataloader.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Arguments passed to the trainer.
    session : int, optional
        The current session.

    Returns
    -------
    Tuple[Any, Any, Any]
        The trainset, trainloader, and testloader for the base classes.
    """
    train_transforms, val_transforms = get_transform(args)

    trainset = dataset_class_map.get(args.dataset, hf_dataset)(
        root=args.dataroot,
        train=True,
        download=True,
        session=session,
        transformations=train_transforms,
        args=args,
    )
    prototypeset = dataset_class_map.get(args.dataset, hf_dataset)(
        root=args.dataroot,
        train=True,
        download=True,
        session=session,
        transformations=val_transforms,
        args=args,
    )
    testset = dataset_class_map.get(args.dataset, hf_dataset)(
        root=args.dataroot,
        train=False,
        download=False,
        session=session,
        transformations=val_transforms,
        args=args,
    )

    trainloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=trainset,  # type: ignore
        batch_size=args.batch_size_base,
        shuffle=True,
        num_workers=args.num_workers,
        sampler=None,
        worker_init_fn=None,
    )
    prototype_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=prototypeset,  # type: ignore
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        worker_init_fn=None,
    )
    testloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=testset,  # type: ignore
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        worker_init_fn=None,
    )

    return prototype_loader, trainloader, testloader
