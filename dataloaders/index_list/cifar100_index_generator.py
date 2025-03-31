"""Generate n-way k-shot index list for CIFAR-100 dataset."""

import os
import random

import torchvision


# Set random seed for reproducibility
random.seed(42)


def load_cifar100(
    start_class_index: int,
    num_classes: int,
    samples_per_class: int,
) -> dict:
    """Load CIFAR-100 dataset.

    Parameters
    ----------
    start_class_index : int
        The starting index of the class.
    num_classes : int
        The number of classes to select.
    samples_per_class : int
        The number of samples to select per class.

    Returns
    -------
    dict
        A dictionary containing the selected indices for each class.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ],
    )

    cifar100_dataset = torchvision.datasets.CIFAR100(
        root="../../data",
        train=True,
        download=True,
        transform=transform,
    )

    # Filter the dataset based on class indices
    selected_classes = list(range(start_class_index, start_class_index + num_classes))
    filtered_dataset = [
        i
        for i in range(len(cifar100_dataset))
        if cifar100_dataset.targets[i] in selected_classes
    ]

    # Randomly select samples from each class
    selected_samples = {}
    for class_idx in selected_classes:
        samples = random.sample(
            [i for i in filtered_dataset if cifar100_dataset.targets[i] == class_idx],
            samples_per_class,
        )
        selected_samples[class_idx] = samples

    return selected_samples


def write_indices_to_file(indices: dict, output_file: str) -> None:
    """Write the selected indices to a file."""
    with open(output_file, "w") as file:
        for _, samples in indices.items():
            for sample_idx in samples:
                file.write(f"{sample_idx}\n")


# Set parameters
global_start_class_index = 0
session_number = 1
num_classes = 10
samples_per_class = 10
output_directory = "cifar100/{}way{}shot".format(num_classes, samples_per_class)
os.makedirs(output_directory, exist_ok=True)

# Load CIFAR-100 dataset
for start_class_index in range(global_start_class_index, 100, num_classes):
    selected_indices = load_cifar100(start_class_index, num_classes, samples_per_class)

    # Write selected indices to different files
    output_file = os.path.join(output_directory, f"session_{session_number}.txt")
    write_indices_to_file(selected_indices, output_file)
    session_number += 1

    print(f"Selected indices written to files in {output_directory}")


# Varify the selected indices
for i in range(session_number, 10):
    with open(f"output_directory/session_{i}.txt", "r") as file:
        indices = file.readlines()

        print(f"Session {i}: {len(indices)} indices selected")
        # load cifar-100 dataset and select the sample with target classes
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ],
        )
        cifar100_dataset = torchvision.datasets.CIFAR100(
            root="../../data",
            train=True,
            download=True,
            transform=transform,
        )
        targets = [cifar100_dataset.targets[idx] for idx in indices]
        print(f"Session {i}: {targets}")
