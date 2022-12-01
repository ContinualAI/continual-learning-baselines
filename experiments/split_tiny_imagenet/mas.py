import torch
from torch.nn import CrossEntropyLoss

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics
)
from avalanche.training.plugins import EvaluationPlugin
from models import MultiHeadVGGSmall
from experiments.utils import set_seed, create_default_args
import avalanche as avl


def mas_stinyimagenet(override_args=None):
    """
    Experiment adapted by
    "A continual learning survey: Defying forgetting in classification tasks"
    by De Lange et al.
    https://doi.org/10.1109/TPAMI.2021.3057446
    """
    args = create_default_args(
        {'cuda': 0, 'lambda_reg': 1., 'alpha': 0.5,
         'verbose': True, 'learning_rate': 0.001,
         'train_mb_size': 200, 'epochs': 50, 'seed': None,
         'dataset_root': None}, override_args)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    """
    "In order to construct a balanced dataset, we assign an equal amount of
    20 randomly chosen classes to each task in a sequence of 10 consecutive
    tasks. This task incremental setting allows using an oracle at test
    time for our evaluation per task, ensuring all tasks are roughly
    similar in terms of difficulty, size, and distribution, making the
    interpretation of the results easier."
    """
    benchmark = avl.benchmarks.SplitTinyImageNet(
        10, return_task_id=True, dataset_root=args.dataset_root)
    model = MultiHeadVGGSmall()
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(
            epoch=True, experience=True, stream=True
        ),
        loss_metrics(
            epoch=True, experience=True, stream=True
        ),
        forgetting_metrics(
            experience=True, stream=True
        ),
        loggers=[interactive_logger])

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    cl_strategy = avl.training.MAS(
        model,
        optimizer,
        criterion, lambda_reg=args.lambda_reg, alpha=args.alpha,
        verbose=args.verbose, train_mb_size=args.train_mb_size,
        train_epochs=args.epochs, eval_mb_size=128, device=device,
        evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = mas_stinyimagenet()
    print(res)
