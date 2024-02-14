import random
from types import SimpleNamespace

import numpy as np
import torch

from avalanche.benchmarks import dataset_benchmark


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args


def restrict_dataset_size(scenario, size: int):
    """
    Util used to restrict the size of the datasets coming from a scenario
    param: size: size of the reduced training dataset
    """
    modified_train_ds = []
    modified_test_ds = []
    modified_valid_ds = []

    if hasattr(scenario, "valid_stream"):
        valid_list = list(scenario.valid_stream)

    for i, train_ds in enumerate(scenario.train_stream):
        train_ds_idx, _ = torch.utils.data.random_split(
            torch.arange(len(train_ds.dataset)),
            (size, len(train_ds.dataset) - size),
        )
        dataset = train_ds.dataset.subset(train_ds_idx)

        modified_train_ds.append(dataset)
        modified_test_ds.append(scenario.test_stream[i].dataset)
        if hasattr(scenario, "valid_stream"):
            modified_valid_ds.append(valid_list[i].dataset)

    scenario = dataset_benchmark(
        modified_train_ds,
        modified_test_ds,
        other_streams_datasets={"valid": modified_valid_ds}
        if len(modified_valid_ds) > 0
        else None,
    )

    return scenario
