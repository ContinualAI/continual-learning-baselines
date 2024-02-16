import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18
from avalanche.training.plugins import EvaluationPlugin, MIRPlugin
from avalanche.training.supervised import Naive, Naive
from experiments.utils import create_default_args, set_seed


def mir_scifar10(override_args=None):
    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 1000,
            "lr": 0.05,
            "train_mb_size": 10,
            "seed": None,
            "subsample": 50,
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
        seed=0,
        fixed_class_order=fixed_class_order,
        train_transform=transforms.ToTensor(),
        eval_transform=transforms.ToTensor(),
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
    )
    scenario = benchmark_with_validation_stream(scenario, 0.05)
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
    res = mir_scifar10()
    print(res)
