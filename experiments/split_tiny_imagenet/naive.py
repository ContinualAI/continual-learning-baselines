import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MultiHeadVGGSmall
from experiments.utils import set_seed, create_default_args


def naive_stinyimagenet(override_args=None):
    """
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this experiment is based on
    "A continual learning survey: Defying forgetting in classification tasks"
    De Lange et. al. (2021).
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9349197

    We use a VGG network, which leads a lower performance than the one from
    De Lange et. al. (2021).
    """
    args = create_default_args({'cuda': 0, 'epochs': 30,
                                'learning_rate': 1e-3, 'train_mb_size': 200, 'seed': None,
                                'dataset_root': None}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitTinyImageNet(
        10, return_task_id=True, dataset_root=args.dataset_root)
    model = MultiHeadVGGSmall(n_classes=200)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.Naive(
        model,
        Adam(model.parameters(), lr=args.learning_rate),
        criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = naive_stinyimagenet()
    print(res)
