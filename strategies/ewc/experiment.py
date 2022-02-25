import unittest
import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MLP
from strategies.utils import create_default_args, get_average_metric, get_target_result


class EWC(unittest.TestCase):
    """
    Reproducing Elastic Weight Consolidation experiments from paper
    "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et. al. (2017).
    https://www.pnas.org/content/114/13/3521
    """

    def test_pmnist(self, override_args=None):
        """Permuted MNIST benchmark"""
        args = create_default_args({'cuda': 0, 'ewc_lambda': 1, 'hidden_size': 1000,
                                    'hidden_layers': 2, 'epochs': 30, 'dropout': 0,
                                    'ewc_mode': 'separate', 'ewc_decay': None,
                                    'learning_rate': 0.001, 'train_mb_size': 256}, override_args)

        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.PermutedMNIST(10)
        model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                    drop_rate=args.dropout)
        criterion = CrossEntropyLoss()

        interactive_logger = avl.logging.InteractiveLogger()

        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        cl_strategy = avl.training.EWC(
            model, SGD(model.parameters(), lr=args.learning_rate), criterion,
            ewc_lambda=args.ewc_lambda, mode=args.ewc_mode, decay_factor=args.ewc_decay,
            train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
            device=device, evaluator=evaluation_plugin)

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"EWC-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('ewc', 'pmnist'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)
