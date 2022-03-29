import unittest
import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.benchmarks.utils.dataset_utils import SubsetWithTargets
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from avalanche.evaluation import metrics as metrics
from models import MultiHeadMLP, SI_CNN
from strategies.utils import create_default_args, get_average_metric, get_target_result, set_seed


def get_cifar_dataset(get_10=True):
    dataset_root = default_dataset_location('cifar10') if get_10 else default_dataset_location('cifar100')
    if get_10:
        train_set = CIFAR10(dataset_root, train=True, download=True)
        test_set = CIFAR10(dataset_root, train=False, download=True)
    else:
        train_set = CIFAR100(dataset_root, train=True, download=True)
        test_set = CIFAR100(dataset_root, train=False, download=True)

    return train_set, test_set


default_cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2762))
])

default_cifar10_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2762))
])


class SynapticIntelligence(unittest.TestCase):
    """
    Reproducing Synaptic Intelligence experiments from paper
    "Continual Learning Through Synaptic Intelligence" by Zenke et. al. (2017).
    http://proceedings.mlr.press/v70/zenke17a.html
    """

    def test_smnist(self, override_args=None):
        """Split MNIST benchmark"""
        args = create_default_args({'cuda': 0, 'si_lambda': 1, 'si_eps': 0.001, 'epochs': 10,
                                    'learning_rate': 0.001, 'train_mb_size': 64, 'seed': 0}, override_args)

        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=True,
                                              fixed_class_order=list(range(10)))
        model = MultiHeadMLP(hidden_size=256, hidden_layers=2)
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

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"SI-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('si', 'smnist'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_pmnist(self, override_args=None):
        """Permuted MNIST benchmark"""
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

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"SI-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('si', 'pmnist'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    # def test_scifar(self, override_args=None):
    #     """Split CIFAR 10/100 benchmark"""
    #     args = create_default_args({'cuda': 0, 'si_lambda': 0.1, 'si_eps': 0.001, 'epochs': 60,
    #                                 'learning_rate': 0.001, 'train_mb_size': 256, 'seed': 0}, override_args)
    #     set_seed(args.seed)
    #     device = torch.device(f"cuda:{args.cuda}"
    #                           if torch.cuda.is_available() and
    #                           args.cuda >= 0 else "cpu")
    #
    #     cifar100_cls = [list(range(i*10, (i+1)*10)) for i in range(5)]
    #     train_cifar10, test_cifar10 = get_cifar_dataset()
    #     train_cifar100, test_cifar100 = get_cifar_dataset(get_10=False)
    #     train_cifar100_list = [SubsetWithTargets(train_cifar100, [i for i, p in enumerate(train_cifar100) if p[1] in el])
    #                            for el in cifar100_cls]
    #     test_cifar100_list = [SubsetWithTargets(test_cifar100, [i for i, p in enumerate(test_cifar100) if p[1] in el])
    #                           for el in cifar100_cls]
    #
    #     train_ds, test_ds, final_classes = avl.benchmarks.utils.concat_datasets_sequentially(
    #         [SubsetWithTargets(train_cifar10, list(range(len(train_cifar10))))] + train_cifar100_list,
    #         [SubsetWithTargets(test_cifar10, list(range(len(test_cifar10))))] + test_cifar100_list)
    #
    #     benchmark = avl.benchmarks.nc_benchmark(train_ds, test_ds, n_experiences=6,
    #                                             task_labels=True,
    #                                             fixed_class_order=list(range(60)),
    #                                             class_ids_from_zero_in_each_exp=True,
    #                                             train_transform=default_cifar10_train_transform,
    #                                             eval_transform=default_cifar10_eval_transform)
    #     model = SI_CNN(hidden_size=512)
    #     criterion = CrossEntropyLoss()
    #
    #     interactive_logger = avl.logging.InteractiveLogger()
    #
    #     evaluation_plugin = avl.training.plugins.EvaluationPlugin(
    #         metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
    #         loggers=[interactive_logger], benchmark=benchmark)
    #
    #     cl_strategy = avl.training.SynapticIntelligence(
    #         model, Adam(model.parameters(), lr=args.learning_rate), criterion,
    #         si_lambda=args.si_lambda, eps=args.si_eps,
    #         train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
    #         device=device, evaluator=evaluation_plugin)
    #
    #     for experience in benchmark.train_stream:
    #         cl_strategy.train(experience)
    #         res = cl_strategy.eval(benchmark.test_stream)
    #
    #     avg_stream_acc = get_average_metric(res)
    #     print(f"SI-SCIFAR Average Stream Accuracy: {avg_stream_acc:.2f}")
    #
    #     target_acc = float(get_target_result('si', 'scifar'))
    #     if args.check:
    #         self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)
