import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MLP
from experiments.utils import set_seed, create_default_args


def lwf_pmnist(override_args=None):
    args = create_default_args({'cuda': 0, 'lwf_alpha': 1, 'lwf_temperature': 1, 'epochs': 5,
                                'layers': 2, 'hidden_size': 1000,
                                'learning_rate': 0.001, 'train_mb_size': 256, 'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.PermutedMNIST(10)
    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.layers)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger], benchmark=benchmark)

    cl_strategy = avl.training.LwF(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        alpha=args.lwf_alpha, temperature=args.lwf_temperature,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res
