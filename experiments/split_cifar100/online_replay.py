import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.supervised import Naive
from experiments.utils import create_default_args, set_seed


def online_replay_scifar100(override_args=None):
    """ 
    Online replay for CIFAR100, the hyperparameters are taken from
    "New Insights on Reducing Abrupt Representation Change in Online Continual Learning", 
    Lucas Caccia et. al., https://openreview.net/forum?id=N8MaByOzUfb 

    Augmentations are not used since they make the results worse in this particular setting
    """

    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 10000,
            "lr": 0.1,
            "train_mb_size": 10,
            "seed": None,
            "batch_size_mem": 10,
        },
        override_args
    )
    set_seed(args.seed)
    fixed_class_order = np.arange(100)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    unique_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
            ),
        ]
    )

    scenario = SplitCIFAR100(
        20,
        return_task_id=False,
        seed=args.seed,
        fixed_class_order=fixed_class_order,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
        train_transform=unique_transform,
        eval_transform=unique_transform,
    )

    scenario = benchmark_with_validation_stream(scenario, 0.05)
    model = SlimResNet18(100)
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
        original_stream=scenario,
        experience_size = args.train_mb_size,
        access_task_boundaries = False
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
    res = online_replay_scifar100()
    print(res)
