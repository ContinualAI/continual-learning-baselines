#!/usr/bin/env python3
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.supervised import OnlineNaive
from experiments.utils import create_default_args, set_seed


def online_replay_scifar10(override_args=None):
    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 1000,
            "lr": 0.1,
            "train_mb_size": 10,
            "seed": 0,
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

    scenario = benchmark_with_validation_stream(scenario, 0.05)
    input_size = (3, 32, 32)
    model = SlimResNet18(10)
    optimizer = SGD(model.parameters(), lr=args.lr)

    interactive_logger = InteractiveLogger()

    loggers = [interactive_logger]

    training_metrics = []

    evaluation_metrics = [
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True, stream=True),
    ]

    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )
    
    storage_policy = ClassBalancedBuffer(args.mem_size, adaptive_size=True)
    plugins = [ReplayPlugin(args.mem_size, storage_policy=storage_policy)]


    cl_strategy = OnlineNaive(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        evaluator=evaluator,
        device=device,
        train_mb_size=args.train_mb_size,
        eval_mb_size=64,
    )

    # For online scenario
    batch_streams = scenario.streams.values()

    for t, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=args.train_mb_size,
            access_task_boundaries=False,
        )

        # Set this to inform strat
        cl_strategy.classes_in_this_experience = experience.classes_in_this_experience

        cl_strategy.train(
            ocl_scenario.train_stream,
            eval_streams=[],
            num_workers=0,
            drop_last=True,
        )

        cl_strategy.eval(scenario.test_stream[: t + 1])

    # Only evaluate at the end on the test stream
    results = cl_strategy.eval(scenario.test_stream)

    return results


if __name__ == "__main__":
    res = online_replay_scifar10()
    print(res)
