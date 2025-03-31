"""Generate few-shot index list for Food101 dataset."""

import argparse
import os
import random

from datasets import load_dataset

from dataloaders.datasets.cub200 import Cub200Dataset
from dataloaders.datasets.miniimagenet import MiniImagenetDataset


def class_count(labels: list[int]) -> dict[int, int]:
    """Count the number of samples per class."""
    classes = set(labels)
    counts = {c: labels.count(c) for c in classes}
    print("Class counts", counts)
    print("Min class count", min(counts.values()))
    return counts


# Food101
dataset_name = "food101"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(dataset_name)

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i
            for i, label in enumerate(dataset["train"]["label"])
            if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# Caltech101
dataset_name = "caltech101"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-caltech101")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# country211
dataset_name = "country211"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_country211")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# country211
dataset_name = "eurosat"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-eurosat")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# country211
dataset_name = "fgvc_aircraft"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_fgvc_aircraft")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# gtsrb
dataset_name = "gtsrb"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_gtsrb")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# oxford_flowers
dataset_name = "oxford_flowers"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("HuggingFaceM4/Oxford-102-Flower")

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i
            for i, label in enumerate(dataset["train"]["label"])
            if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# oxford_flowers
dataset_name = "oxford_pets"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-pets")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# resisc45
dataset_name = "resisc45"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-resisc45")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# resisc45
dataset_name = "stanford_cars"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_cars")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# voc2007
dataset_name = "voc2007"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_voc2007")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# dtd
dataset_name = "dtd"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(
        "HuggingFaceM4/DTD_Describable-Textures-Dataset",
        "partition_1",
    )

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i
            for i, label in enumerate(dataset["train"]["label"])
            if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# sun397
dataset_name = "sun397"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(
        "HuggingFaceM4/sun397",
        "standard-part1-120k",
    )

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i
            for i, label in enumerate(dataset["train"]["label"])
            if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# CUB_200_2011
dataset_name = "cub200"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    args = argparse.Namespace()
    dataset = Cub200Dataset("data/CUB_200_2011", args)

    num_classes = len(set(dataset.labels))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset.labels)
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    # make sure of include the samples from FSCIL literature
    fscil_samples = []
    for i in range(2, 12):
        txt_path = f"dataloaders/index_list/cub200/session_{i}.txt"
        with open(txt_path) as f:
            fscil_samples.extend(f.read().splitlines())
    fscil_sample_indices = {dataset.images.index(i) for i in fscil_samples}

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [i for i, label in enumerate(dataset.labels) if label == class_index]
        # first 5 samples from FSCIL indices
        selected_samples[class_index] = list(
            fscil_sample_indices.intersection(set(indices)),
        )
        # remove already selected samples
        indices = list(set(indices) - fscil_sample_indices)
        # select rest of the samples randomly
        selected_samples[class_index].extend(
            random.sample(
                indices,
                samples_per_class - len(selected_samples[class_index]),
            ),
        )

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# mini_imagenet
dataset_name = "mini_imagenet"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    args = argparse.Namespace()
    dataset = MiniImagenetDataset("data/miniimagenet", args)

    num_classes = len(set(dataset.labels))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset.labels)
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    # make sure of include the samples from FSCIL literature
    fscil_samples = []
    for i in range(2, 10):
        txt_path = f"dataloaders/index_list/mini_imagenet/session_{i}.txt"
        with open(txt_path) as f:
            fscil_samples.extend(f.read().splitlines())
    fscil_samples = [name.split("/")[-1] for name in fscil_samples]
    fscil_sample_indices = {dataset.images.index(i) for i in fscil_samples}

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [i for i, label in enumerate(dataset.labels) if label == class_index]
        # first 5 samples from FSCIL indices
        selected_samples[class_index] = list(
            fscil_sample_indices.intersection(set(indices)),
        )
        # remove already selected samples
        indices = list(set(indices) - fscil_sample_indices)
        # select rest of the samples randomly
        selected_samples[class_index].extend(
            random.sample(
                indices,
                samples_per_class - len(selected_samples[class_index]),
            ),
        )

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")

# cifar100
dataset_name = "cifar100"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(dataset_name)["train"]

    num_classes = len(set(dataset["fine_label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["fine_label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    # make sure of include the samples from FSCIL literature
    fscil_samples = []
    for i in range(2, 10):
        txt_path = f"dataloaders/index_list/cifar100/5way5shot/session_{i}.txt"
        with open(txt_path) as f:
            fscil_samples.extend(f.read().splitlines())
    fscil_sample_indices = {int(name) for name in fscil_samples}  # type: ignore

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["fine_label"]) if label == class_index
        ]

        # first 5 samples from FSCIL indices
        selected_samples[class_index] = list(
            fscil_sample_indices.intersection(set(indices)),
        )

        # remove already selected samples
        indices = list(set(indices) - fscil_sample_indices)

        # select rest of the samples randomly
        selected_samples[class_index].extend(
            random.sample(
                indices,
                samples_per_class - len(selected_samples[class_index]),
            ),
        )

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")
