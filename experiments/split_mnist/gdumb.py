import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MLP
from experiments.utils import set_seed, create_default_args


def gdumb_smnist(override_args=None):
    """
    "GDumb: A Simple Approach that Questions Our Progress in Continual Learning" by Prabhu et. al. (2020).
    https://link.springer.com/chapter/10.1007/978-3-030-58536-5_31
    """
    args = create_default_args({'cuda': 0, 'hidden_size': 400, 'mem_size': 4400,
                                'hidden_layers': 2, 'epochs': 10, 'dropout': 0,
                                'learning_rate': 0.1, 'train_mb_size': 16, 'seed': None}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)
    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                drop_rate=args.dropout, relu_act=True)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.GDumb(
        model, SGD(model.parameters(), lr=args.learning_rate), criterion,
        mem_size=args.mem_size,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = gdumb_smnist()
    print(res)
