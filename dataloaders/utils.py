"""Dataloader utilities."""

import argparse

import numpy as np


def get_session_classes(args: argparse.Namespace, session: int) -> np.ndarray:
    """Get the classes for the current session.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the trainer.
    session : int
        The current session.

    Returns
    -------
    np.ndarray
        The classes for the current session.
    """
    session_start_class, session_end_class = 0, args.base_class
    if session > 0:
        session_start_class = args.base_class + (session - 1) * args.way
        session_end_class = session_start_class + args.way
    return np.arange(session_start_class, session_end_class)


def get_few_shot_samples_indices_per_class(
    dataset: str,
    classes: np.ndarray,
    shot: int = 0,
) -> list:
    """Get few-shot samples indices per class.

    Parameters
    ----------
    dataset : str
        The dataset name.

    Returns
    -------
    dict
        Few-shot samples indices per class.
    """
    if shot == 0:
        raise ValueError("Shot cannot be 0.")

    # Add a return statement that returns an empty dictionary
    sample_ids: dict = {}
    with open("dataloaders/index_list/{}_index_list.txt".format(dataset), "r") as file:
        for line in file:
            class_index, index = line.strip().split()
            if int(class_index) not in sample_ids:
                sample_ids[int(class_index)] = []
            sample_ids[int(class_index)].append(int(index))

    # select samples of classes of current session specified in 'classes'
    sample_ids = {key: val for key, val in sample_ids.items() if key in classes}

    select_samples = []
    for key in sample_ids:
        select_samples.extend(sample_ids[key][:shot])

    return select_samples
