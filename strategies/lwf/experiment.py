import unittest
import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from avalanche.evaluation import metrics as metrics
from models import MLP, MultiHeadVGGSmall
from strategies.utils import create_default_args, get_average_metric, get_target_result


class LwF(unittest.TestCase):
    """
    Reproducing Learning without Forgetting. Original paper is
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this class reproduces LwF experiments
    on Split MNIST and Permuted MNIST from
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf
    We managed to surpass the performances reported in the paper by slightly
    changing the model architecture or the training hyperparameters.
    Experiments on Tiny Image Net are taken from
    "A continual learning survey: Defying forgetting in classification tasks" De Lange et. al. (2021).
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9349197
    """

    def test_smnist(self, override_args=None):
        """Split MNIST benchmark"""
        args = create_default_args({'cuda': 0, 'lwf_alpha': 1,
                                    'lwf_temperature': 1, 'epochs': 10,
                                    'layers': 1, 'hidden_size': 256,
                                    'learning_rate': 0.001, 'train_mb_size': 128}, override_args)

        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)
        model = MLP(hidden_size=args.hidden_size, hidden_layers=args.layers)
        criterion = CrossEntropyLoss()

        interactive_logger = avl.logging.InteractiveLogger()

        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        cl_strategy = avl.training.LwF(
            model, SGD(model.parameters(), lr=args.learning_rate), criterion,
            alpha=args.lwf_alpha, temperature=args.lwf_temperature,
            train_mb_size=args.train_mb_size, train_epochs=args.epochs,
            device=device, evaluator=evaluation_plugin)

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"LwF-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lwf', 'smnist'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)

    def test_pmnist(self, override_args=None):
        """Permuted MNIST benchmark"""
        args = create_default_args({'cuda': 0, 'lwf_alpha': 1, 'lwf_temperature': 1, 'epochs': 5,
                                    'layers': 2, 'hidden_size': 1000,
                                    'learning_rate': 0.001, 'train_mb_size': 256}, override_args)

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

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"LwF-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lwf', 'pmnist'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)

    def test_stinyimagenet(self, override_args=None, dataset_root=None):
        """Split Tiny ImageNet benchmark"""
        args = create_default_args({'cuda': 0,
                                    'lwf_alpha': 10, 'lwf_temperature': 2, 'epochs': 70,
                                    'learning_rate': 0.0001, 'train_mb_size': 200}, override_args)

        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.SplitTinyImageNet(
            10, return_task_id=True, dataset_root=dataset_root)
        model = MultiHeadVGGSmall(n_classes=20)
        criterion = CrossEntropyLoss()

        interactive_logger = avl.logging.InteractiveLogger()

        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        cl_strategy = avl.training.LwF(
            model, SGD(model.parameters(), lr=args.learning_rate, momentum=0.9), criterion,
            alpha=args.lwf_alpha, temperature=args.lwf_temperature,
            train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
            device=device, evaluator=evaluation_plugin)

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"LwF-SplitTinyImageNet Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lwf', 'stiny-imagenet'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)
