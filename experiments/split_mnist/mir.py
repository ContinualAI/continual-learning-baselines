import numpy as np
import torch
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin, MIRPlugin
from avalanche.training.supervised import Naive
from experiments.utils import create_default_args, set_seed, restrict_dataset_size


def mir_smnist(override_args=None):
    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 500,
            "lr": 0.05,
            "train_mb_size": 10,
            "seed": None,
            "subsample": 50,
            "batch_size_mem": 10,
            "dataset_size": 1000,
        },
        override_args
    )
    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    scenario = SplitMNIST(
        5,
        return_task_id=False,
        seed=0,
        train_transform=None,
        eval_transform=None,
    )
    scenario = benchmark_with_validation_stream(scenario, 0.05)
    scenario = restrict_dataset_size(scenario, args.dataset_size)
    model = SimpleMLP(10, hidden_size=400, hidden_layers=1)
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
    plugins = [
        MIRPlugin(
            mem_size=args.mem_size, subsample=args.subsample, batch_size_mem=args.batch_size_mem
        )
    ]
    cl_strategy = Naive(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        evaluator=evaluator,
        device=device,
        train_mb_size=args.train_mb_size,
        eval_mb_size=64,
    )
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
    res = mir_smnist()
    print(res)
