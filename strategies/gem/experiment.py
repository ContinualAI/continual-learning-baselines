import unittest
import avalanche as avl
import torch
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MLP, MultiHeadReducedResNet18
from strategies.utils import create_default_args, get_average_metric, get_target_result, set_seed


class GEM_reduced(avl.training.GEM):
    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """Select only 1000 patterns for each experience as in GEM paper."""
        self.dataloader = TaskBalancedDataLoader(
            AvalancheSubset(self.adapted_dataset, indices=list(range(1000))),
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory)


class GEM(unittest.TestCase):
    """
    Reproducing GEM experiments from paper
    "Gradient Episodic Memory for Continual Learning" by Lopez-paz et. al. (2017).
    https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html
    """

    def test_pmnist(self, override_args=None):
        """Permuted MNIST benchmark"""
        args = create_default_args({'cuda': 0, 'patterns_per_exp': 1000, 'hidden_size': 100,
                                    'hidden_layers': 2, 'epochs': 1, 'dropout': 0,
                                    'mem_strength': 0.5,
                                    'learning_rate': 0.1, 'train_mb_size': 10, 'seed': 0}, override_args)
        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.PermutedMNIST(20)
        model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                    drop_rate=args.dropout)
        criterion = CrossEntropyLoss()

        interactive_logger = avl.logging.InteractiveLogger()

        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        cl_strategy = GEM_reduced(
            model, SGD(model.parameters(), lr=args.learning_rate), criterion,
            patterns_per_exp=args.patterns_per_exp, memory_strength=args.mem_strength,
            train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
            device=device, evaluator=evaluation_plugin)

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"GEM-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('gem', 'pmnist'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_scifar100(self, override_args=None):
        """Split CIFAR-100 benchmark"""
        args = create_default_args({'cuda': 0, 'patterns_per_exp': 256, 'epochs': 1,
                                    'mem_strength': 0.5, 'learning_rate': 0.1, 'train_mb_size': 10,
                                    'seed': 0}, override_args)
        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.SplitCIFAR100(20, return_task_id=True)
        model = MultiHeadReducedResNet18()
        criterion = CrossEntropyLoss()

        interactive_logger = avl.logging.InteractiveLogger()

        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        cl_strategy = avl.training.GEM(
            model, SGD(model.parameters(), lr=args.learning_rate, momentum=0.9), criterion,
            patterns_per_exp=args.patterns_per_exp, memory_strength=args.mem_strength,
            train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
            device=device, evaluator=evaluation_plugin)

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"GEM-SCIFAR100 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('gem', 'scifar100'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
