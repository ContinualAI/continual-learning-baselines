#!/usr/bin/env python3
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, OnlineER_ACE
from experiments.utils import create_default_args, set_seed


def eracl_scifar10(override_args=None):
    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 1000,
            "lr": 0.1,
            "train_mb_size": 10,
            "seed": None,
            "batch_size_mem": 10,
        },
        override_args
    )
    set_seed(args.seed)
    fixed_class_order = np.arange(10)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    scenario = SplitCIFAR10(
        5,
        return_task_id=False,
        seed=args.seed,
        fixed_class_order=fixed_class_order,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
    )

    model = SlimResNet18(1)
    model.linear = IncrementalClassifier(model.linear.in_features, 1)
    optimizer = SGD(model.parameters(), lr=args.lr)

    interactive_logger = InteractiveLogger()

    loggers = [interactive_logger]

    training_metrics = []

    evaluation_metrics = [
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True, stream=True),
    ]

    # Create main evaluator that will be used by the training actor
    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )

    plugins = []

    #######################
    #  Strategy Creation  #
    #######################

    cl_strategy = OnlineER_ACE(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        evaluator=evaluator,
        device=device,
        train_mb_size=args.train_mb_size,
        eval_mb_size=64,
        mem_size=args.mem_size,
        batch_size_mem=args.batch_size_mem,
    )

    ###################
    #  TRAINING LOOP  #
    ###################

    print("Starting experiment...")

    print([p.__class__.__name__ for p in cl_strategy.plugins])

    ocl_scenario = split_online_stream(
        original_stream=scenario.train_stream,
        experience_size=10,
        access_task_boundaries=False,
    )
    for t, experience in enumerate(ocl_scenario):
        cl_strategy.train(
            experience,
            eval_streams=[],
            num_workers=0,
            drop_last=True,
        )

    results = cl_strategy.eval(scenario.test_stream)

    return results


if __name__ == "__main__":
    res = eracl_scifar10()
    print(res)
