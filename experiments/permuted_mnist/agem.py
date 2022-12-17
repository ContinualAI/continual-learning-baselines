import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from avalanche.evaluation import metrics as metrics
from models import MLP
from experiments.utils import set_seed, create_default_args


def agem_pmnist(override_args=None):
    """
    "Efficient Lifelong Learning with A-GEM" by Chaudhry et. al. (2019).
    https://openreview.net/pdf?id=Hkf2_sC5FX
    """
    args = create_default_args({'cuda': 0, 'patterns_per_exp': 250, 'hidden_size': 256,
                                'hidden_layers': 2, 'epochs': 1, 'dropout': 0,
                                'sample_size': 256,
                                'learning_rate': 0.1, 'train_mb_size': 10,
                                'seed': None}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.PermutedMNIST(17)
    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                drop_rate=args.dropout)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.AGEM(
        model, SGD(model.parameters(), lr=args.learning_rate), criterion,
        patterns_per_exp=args.patterns_per_exp, sample_size=args.sample_size,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None 
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)
    return res


if __name__ == '__main__':
    res = agem_pmnist()
    print(res)
