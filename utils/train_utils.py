"""Contains a collection of utility functions for the project."""

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from utils.constants import (
    dataset_roots,
    dataset_specific_configs,
    fscil_base_classes,
    fscil_ways,
    fscit_base_classes,
    fscit_ways,
    num_classes,
)


def sanity_check(args: argparse.Namespace) -> None:
    """Sanity check for the command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    None
    """
    if args.dataset not in ["cifar100", "cub200", "mini_imagenet"]:
        assert (
            args.fsl_setup == "FSCIT"
        ), "FSCIL is only supported for CIFAR-100, CUB-200, and miniImageNet"


def override_training_configs(args: argparse.Namespace) -> argparse.Namespace:
    """Override the training configurations.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    argparse.Namespace: The command-line arguments
    with the overridden training configurations.
    """
    args.exp_name = args.dataset + "_" + args.exp_name
    if isinstance(args.adapt_blocks, int):
        if args.adapt_blocks < 0:
            args.adapt_blocks *= -1
            args.adapt_blocks = list(range(args.adapt_blocks, 12))
        else:
            args.adapt_blocks = list(range(args.adapt_blocks))

    args.save_path = os.path.join("checkpoint", args.exp_name)

    args.size_crops = [224, 224]
    if "384" in args.hf_model_checkpoint:
        args.size_crops = [384, 384]
    return args


def get_dataset_configs(args: argparse.Namespace) -> argparse.Namespace:
    """Set up the datasets.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    argparse.Namespace: The command-line arguments
    with the datasets configs.
    """
    if args.fsl_setup == "FSCIL":
        args.base_class = fscil_base_classes[args.dataset]
        args.way = fscil_ways[args.dataset]
    else:
        args.base_class = fscit_base_classes[args.dataset]
        args.way = fscit_ways[args.dataset]
    args.dataroot = dataset_roots.get(args.dataset, "./data")
    args.num_classes = num_classes[args.dataset]
    args.sessions = ((args.num_classes - args.base_class) // args.way) + 1
    dataset_specific_configs_subset = dataset_specific_configs.get(
        args.hf_model_checkpoint,
        {},
    )
    if dataset_specific_configs_subset:
        for key, value in (
            dataset_specific_configs_subset[args.fsl_setup].get(args.dataset, {}).items()  # type: ignore
        ):
            setattr(args, key, value)

    return args


def str2bool(v: str) -> bool:
    """Convert a string to a boolean value.

    Parameters
    ----------
    v : str
        The string to convert.

    Returns
    -------
    bool: The boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_command_line_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    """Get the command line parser for the script.

    Returns
    -------
        argparse.ArgumentParser: The command line parser.
    """
    parser = argparse.ArgumentParser()

    # about experiment and dataset
    parser.add_argument(
        "--exp_name",
        type=str,
        default="baseline",
        help="experiment name used for saving and logging",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=num_classes.keys(),
    )
    parser.add_argument("--dataroot", type=str, default="./data")
    parser.add_argument(
        "--result_key",
        type=str,
        default="",
        help="Additional key to filename for saving results at a different file.",
    )

    # about pre-training
    parser.add_argument(
        "--batch_size_base",
        type=int,
        default=64,
        help="batch size for base training session",
    )
    parser.add_argument(
        "--epochs_base",
        type=int,
        default=100,
        help="number of epochs for base training session",
    )
    parser.add_argument(
        "--lr_base",
        type=float,
        default=0.1,
        help="learning rate for base training session",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="Cosine",
        choices=["Step", "Milestone", "Cosine"],
        help="learning rate schedule for base training session",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[60, 70],
        help="milestone epochs for learning rate schedule",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=40,
        help="step size for learning rate schedule",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.0005,
        help="weight decay for base training session",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for base training session",
    )
    parser.add_argument(
        "--ce_loss_factor",
        type=float,
        default=1.0,
        help="coefficient of the cross-entropy loss",
    )

    # for contrastive learning
    parser.add_argument(
        "--moco_dim",
        default=128,
        type=int,
        help="feature dimension of the model output (default: 128)",
    )
    parser.add_argument(
        "--moco_k",
        default=65536,
        type=int,
        help="queue size; number of negative keys (default: 65536)",
    )
    parser.add_argument(
        "--moco_m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco_t",
        default=0.07,
        type=float,
        help="softmax temperature (default: 0.07)",
    )
    parser.add_argument(
        "--moco_loss_factor",
        type=float,
        default=1.0,
        help="coefficient of the moco loss; disabled by default",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=2,
        help="number of views for contrastive learning",
    )
    parser.add_argument(
        "--ce_loss_factor_incft",
        type=float,
        default=1.0,
        help="coefficient of the cross-entropy loss for incremental fine-tuning",
    )

    # model config
    parser.add_argument(
        "--hf_model_checkpoint",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="name of the huggingface model checkpoint to load",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="vit-b16",
        help="encoder architecture",
    )

    parser.add_argument(
        "--num_mlp",
        type=int,
        default=0,
        help="number of mlp layers in projection head, no mlp by default",
    )

    # few-shot configs
    parser.add_argument(
        "--shot",
        type=int,
        default=-1,
        help="number of shots; -1 means shots taken as the defaults for the dataset",
    )
    parser.add_argument(
        "--fsl_setup",
        type=str,
        default="FSCIT",
        choices=["FSCIT", "FSCIL"],
        help="Few-shot learning setup, FSCIL comes with large base, while FSCIT"
        " is n-way k-shot base setting; Only 3 datasets support FSCIL that are "
        "reported by exising literaure: CIFAR-100, CUB-200, and miniImageNet.",
    )

    # incremental fine-tune configs
    parser.add_argument(
        "--incft",
        type=str2bool,
        default=False,
        help="incrmental finetuning",
    )
    parser.add_argument(
        "--incft_layers",
        type=str,
        default="classifier",
        choices=["classifier", "pet", "classifier+pet"],
        help="layers to be fine-tuned in incremental fine-tuning",
    )
    parser.add_argument(
        "--epochs_incremental",
        type=int,
        default=0,
        help="number of epochs for incremental fine-tuning",
    )
    parser.add_argument(
        "--consistency_loss_factor",
        type=float,
        default=1.0,
        help="coefficient of the consistency loss; disabled by default",
    )
    parser.add_argument(
        "--inc_ft_lr_factor",
        type=float,
        default=1.0,
        help="Factor to scale the incremental fine-tuning learning rate",
    )

    # test configs
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=15,
        help="evaluation frequency",
    )

    # about training
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="number of random seeds",
    )

    # PET specific configs
    parser.add_argument(
        "--pet_cls",
        type=str,
        default=None,
        choices=[None, "Prefix", "Adapter", "LoRA"],
    )
    parser.add_argument("--rank", type=int, default=5)

    # FSCIT configs
    parser.add_argument(
        "--encoder_ft_start_layer",
        type=int,
        default=500,
        help="Encoder fine-tuning start layer; -1 means full-tuning; Use large number of freeze the whole network.",
    )
    parser.add_argument("--adapt_blocks", type=int, default=0)
    parser.add_argument(
        "--encoder_ft_start_epoch",
        type=int,
        default=0,
        help="Epoch at which to start fine-tuning the encoder",
    )
    parser.add_argument(
        "--encoder_lr_factor",
        type=float,
        default=1,
        help="Factor to scale the pre-trained encoder learning rate",
    )
    parser.add_argument(
        "--limited_base_class",
        type=int,
        default=-1,
        help="Number of base classes to use for FSCIT",
    )
    parser.add_argument(
        "--reduced_base_shot",
        type=int,
        default=-1,
        help="Reduced base shot. -1 indicate disabled",
    )
    parser.add_argument(
        "--pet_on_teacher",
        type=str2bool,
        default=False,
        help="Flag to set whether or not use PET on the teacher model. If not, "
        "teacher will be different from student, and will only be updated for "
        "the pre-trained parameters that are opt for fine-tuning. If no pre-trained"
        " parameters are tuned, EMA will enb up remaining the same as the pre-trained model.",
    )
    parser.add_argument(
        "--update_base_classifier_with_prototypes",
        type=str2bool,
        default=False,
        help="Flag to set whether or not update the base classifier with the prototypes",
    )
    parser.add_argument(
        "--start_training_with_prototypes",
        type=str2bool,
        default=False,
        help="Flag to set whether or not update the base classifier with the prototypes "
        "before the start of base training. Bascially, this is used to avoid training "
        "from randomly initialized classifier.",
    )
    parser.add_argument(
        "--add_bias_in_classifier",
        type=str2bool,
        default=False,
        help="Flag to set whether or not add bias in the classifier layer",
    )
    return parser


def calculate_cdf(prototypes: List[np.ndarray]) -> None:
    """Create a list to store inter-class distances.

    Parameters
    ----------
    prototypes : list
        A list of class prototypes,
        where each prototype is a 1D numpy array.

    Returns
    -------
    None
    """
    inter_class_distances = []

    # Calculate inter-class distances for all pairs of prototypes
    for i in range(len(prototypes)):
        for j in range(i + 1, len(prototypes)):
            cosine_sim = cosine_similarity([prototypes[i]], [prototypes[j]])[0][0]
            inter_class_distance = 1 - cosine_sim
            inter_class_distances.append(inter_class_distance)

    # Sort the distances
    sorted_distances = np.sort(inter_class_distances)

    # Calculate the cumulative distances
    cumulative_distances = np.arange(1, len(sorted_distances) + 1) / len(
        sorted_distances,
    )

    # Plot the CDF
    plt.plot(sorted_distances, cumulative_distances)
    plt.xlabel("Inter-class Distance")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distance Function (CDF) for 100 Class Prototypes")
    plt.grid()
    plt.show()


def pseudo_intra_class_cdf() -> None:
    """Calculate intra-class distances and plot cumulative distance function (CDF)."""
    # Sample embeddings and their corresponding labels
    embeddings = np.random.rand(
        100,
        4,
    )  # Assuming 100 samples with 4-dimensional embeddings
    labels = np.random.randint(0, 10, size=100)  # Assuming 10 classes

    # Prototypes for each class (assuming 10 classes and 4-dimensional prototypes)
    prototypes = [np.random.rand(4) for _ in range(10)]

    # Create a dictionary to store intra-class distances for each class
    intra_class_distances: Dict[int, List[float]] = {
        class_label: [] for class_label in set(labels)
    }

    # Calculate intra-class distances for each sample
    for i in range(len(embeddings)):
        sample = embeddings[i]
        class_label = labels[i]
        prototype = prototypes[class_label]

        # Calculate cosine similarity
        cosine_sim = cosine_similarity([sample], [prototype])[0][0]

        # Calculate intra-class distance
        intra_class_distance = 1 - (cosine_sim / len(prototypes))

        # Append to the corresponding class's list
        intra_class_distances[class_label].append(intra_class_distance)

    for key, value in intra_class_distances.items():
        # average
        intra_class_distances[key] = np.mean(value)  # type: ignore

    distances = intra_class_distances.values()
    sorted_distances = np.sort(list(distances))
    cumulative_distances = np.arange(1, len(prototypes) + 1) / len(prototypes)
    plt.plot(sorted_distances, cumulative_distances, label="Overall")

    plt.xlabel("Intra-class Distance")
    plt.ylabel("Cumulative Probability")
    plt.title("Intra-class Cumulative Distance Function (CDF)")
    plt.legend()
    plt.grid()
    plt.show()


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed.

    Returns
    -------
        None
    """
    if seed == 0:
        print(" random seed")
        torch.backends.cudnn.benchmark = True
    else:
        print("manual seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_path(path: str) -> None:
    """Ensure the path exists.

    Parameters
    ----------
    path : str
        The path to ensure.

    Returns
    -------
    None
    """
    os.makedirs(path, exist_ok=True)


class Averager:
    """A to calculate the average of a series of numbers."""

    def __init__(self) -> None:
        """Initialize the Averager class."""
        self.n = 0.0  # Change the type of self.n from int to float
        self.v = 0.0

    def add(self, x: float) -> None:
        """Add a number to the series.

        Parameters
        ----------
        x : float
            The number to add.

        Returns
        -------
        None
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self) -> float:
        """Return the average of the series.

        Returns
        -------
        float: The average of the series.
        """
        return self.v


def count_acc(logits: torch.Tensor, label: torch.Tensor) -> float:
    """Calculate the accuracy of the model.

    Parameters
    ----------
    logits : torch.Tensor
        The model's predictions.
    label : torch.Tensor
        The true labels.

    Returns
    -------
    float: The accuracy of the model.
    """
    pred = torch.argmax(logits, dim=1)
    return (pred == label).float().mean().item()


def save_results(results_dict: dict, args: argparse.Namespace) -> None:
    """Save the results to a file.

    Parameters
    ----------
    results_dict : dict
        The results dictionary.
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    None
    """
    ensure_path("results")
    path = "results/{}_{}_{}_{}.tsv".format(
        args.dataset,
        args.fsl_setup,
        args.result_key,
        args.num_seeds,
    )
    args_dict = dict(args.__dict__)  # type: ignore
    if not os.path.exists(path):
        with open(path, "w") as f:
            output = (
                [k + "_last" for k in list(results_dict.keys())]
                + list(results_dict.keys())
                + list(args_dict.keys())
            )
            f.write("\t".join(output) + "\n")

    with open(path, "a") as f:
        output_res = (
            [str(results_dict[key][-1]) for key in results_dict]
            + [str(results_dict[key]) for key in results_dict]
            + [str(args_dict[key]) for key in args_dict]
        )
        f.write("\t".join(output_res) + "\n")
