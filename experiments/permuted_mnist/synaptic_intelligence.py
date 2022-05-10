import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MultiHeadMLP
from experiments.utils import set_seed, create_default_args


def synaptic_intelligence_pmnist(override_args=None):
    args = create_default_args({'cuda': 0, 'si_lambda': 0.1, 'si_eps': 0.1, 'epochs': 20,
                                'learning_rate': 0.001, 'train_mb_size': 256, 'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.PermutedMNIST(10)
    model = MultiHeadMLP(hidden_size=2000, hidden_layers=2)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger], benchmark=benchmark)

    cl_strategy = avl.training.SynapticIntelligence(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        si_lambda=args.si_lambda, eps=args.si_eps,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = synaptic_intelligence_pmnist()
    print(res)
