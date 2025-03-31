"""Load food101 dataset."""

import argparse
from typing import Any

from datasets import load_dataset

from dataloaders.utils import (
    get_few_shot_samples_indices_per_class,
    get_session_classes,
)


hf_dataset_name_map = {
    "food101": "food101",
    "caltech101": "clip-benchmark/wds_vtab-caltech101",
    "country211": "clip-benchmark/wds_country211",
    "eurosat": "clip-benchmark/wds_vtab-eurosat",
    "fgvc_aircraft": "clip-benchmark/wds_fgvc_aircraft",
    "gtsrb": "clip-benchmark/wds_gtsrb",
    "oxford_flowers": "HuggingFaceM4/Oxford-102-Flower",
    "oxford_pets": "clip-benchmark/wds_vtab-pets",
    "resisc45": "clip-benchmark/wds_vtab-resisc45",
    "stanford_cars": "clip-benchmark/wds_cars",
    "voc2007": "clip-benchmark/wds_voc2007",
    "dtd": "HuggingFaceM4/DTD_Describable-Textures-Dataset",
    "sun397": "HuggingFaceM4/sun397",
    "cifar100": "cifar100",
    "omniglot": "omniglot",
    "vgg-flowers": "vgg_flowers",
    "ucf101": "ucf101",
}

additional_hf_configs = {"dtd": ["partition_1"], "sun397": ["standard-part1-120k"]}


def get_hf_data(dataset_name: str, split: str) -> Any:
    """Get data from Hugging Face dataset."""
    dataset = load_dataset(
        hf_dataset_name_map.get(dataset_name),
        *additional_hf_configs.get(dataset_name, []),
    )

    if split == "validation" and split not in dataset:
        if "test" in dataset:
            split = "test"
        elif "valid" in dataset:
            split = "valid"
    dataset = dataset[split]

    if dataset_name in [
        "caltech101",
        "eurosat",
        "gtsrb",
        "oxford_pets",
        "resisc45",
        "voc2007",
    ]:
        dataset = dataset.remove_columns(["__key__", "__url__"])
        dataset = dataset.rename_column("cls", "label")
        dataset = dataset.rename_column("webp", "image")

    if dataset_name in ["country211", "fgvc_aircraft", "stanford_cars"]:
        dataset = dataset.remove_columns(["__key__", "__url__"])
        dataset = dataset.rename_column("cls", "label")
        dataset = dataset.rename_column("jpg", "image")

    if dataset_name in ["cifar100"]:
        dataset = dataset.remove_columns(["coarse_label"])
        dataset = dataset.rename_column("fine_label", "label")
        dataset = dataset.rename_column("img", "image")

    return dataset


def hf_dataset(
    root: str,
    args: argparse.Namespace,
    train: bool = True,
    download: bool = False,
    session: int = 0,
    transformations: Any = None,
) -> Any:
    """Initialize HF dataset.

    Parameters
    ----------
    root : str
        Root directory of dataset
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
    split = "train" if train else "validation"
    dataset = get_hf_data(args.dataset, split)

    if session == -1:  # way of getting the whole dataset (not needed for this project)
        return dataset

    classes_at_current_session = get_session_classes(args, session)

    if train:  # few-shot training samples
        if args.fsl_setup == "FSCIL" and session == 0 and args.reduced_base_shot < 0:
            sample_ids = [
                i
                for i, label in enumerate(dataset["label"])
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
                args.dataset,
                classes_at_current_session,
                shot,
            )
    else:  # validation; all samples of the classes till curr session
        sample_ids = [
            i
            for i, label in enumerate(dataset["label"])
            if label <= max(classes_at_current_session)
        ]
    dataset = dataset.select(sample_ids)

    def multi_view_transform(image: Any) -> Any:
        """Apply multi-view transformation."""
        return [transformations(image) for _ in range(args.num_views)]

    def preprocess_train(example_batch: Any) -> Any:
        """Apply train_transforms across a batch."""
        example_batch["image"] = [
            multi_view_transform(image.convert("RGB"))
            for image in example_batch["image"]
        ]
        return example_batch

    dataset.set_transform(preprocess_train)

    return dataset
