import unittest
import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MLP, MultiHeadReducedResNet18
from strategies.utils import create_default_args, get_average_metric, get_target_result


class AGEM(unittest.TestCase):
    """
    Reproducing Average-GEM experiments from paper
    "Efficient Lifelong Learning with A-GEM" by Chaudhry et. al. (2019).
    https://openreview.net/pdf?id=Hkf2_sC5FX
    The main difference with the original paper is that we do not append any task descriptor
    to the model input.
    We train on the last 17 experiences since we apply the evaluation protocol defined
    in the paper but we do not perform model selection.
    """

    def test_pmnist(self, override_args=None):
        """Permuted MNIST benchmark"""
        args = create_default_args({'cuda': 0, 'patterns_per_exp': 250, 'hidden_size': 256,
                                    'hidden_layers': 2, 'epochs': 1, 'dropout': 0,
                                    'sample_size': 256,
                                    'learning_rate': 0.1, 'train_mb_size': 10}, override_args)

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
            loggers=[interactive_logger], benchmark=benchmark)

        cl_strategy = avl.training.AGEM(
            model, SGD(model.parameters(), lr=args.learning_rate), criterion,
            patterns_per_exp=args.patterns_per_exp, sample_size=args.sample_size,
            train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
            device=device, evaluator=evaluation_plugin)

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"AGEM-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('agem', 'pmnist'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)

    def test_scifar100(self, override_args=None):
        """Split CIFAR-100 benchmark"""
        args = create_default_args({'cuda': 0, 'patterns_per_exp': 65, 'epochs': 1,
                                    'sample_size': 1300, 'learning_rate': 0.03, 'train_mb_size': 10}, override_args)

        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.SplitCIFAR100(17, return_task_id=True, fixed_class_order=list(range(85)))
        model = MultiHeadReducedResNet18()
        criterion = CrossEntropyLoss()

        interactive_logger = avl.logging.InteractiveLogger()

        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        cl_strategy = avl.training.AGEM(
            model, SGD(model.parameters(), lr=args.learning_rate, momentum=0.9), criterion,
            patterns_per_exp=args.patterns_per_exp, sample_size=args.sample_size,
            train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
            device=device, evaluator=evaluation_plugin)

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"AGEM-SCIFAR100 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('agem', 'scifar100'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)
