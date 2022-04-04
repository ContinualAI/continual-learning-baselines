import unittest
import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss

from avalanche.evaluation import metrics as metrics
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.plugins import ReplayPlugin

from strategies.utils import create_default_args, get_average_metric, \
    get_target_result, set_seed
from models.lamaml_cnns import MTConvCIFAR, MTConvTinyImageNet


class LaMAML(unittest.TestCase):
    """
    Reproducing LaMAML. Original paper is
    "La-MAML: Look-ahead Meta Learning for Continual Learning"
    by Gupta et. al. (2020).
    https://arxiv.org/abs/2007.13904

    """

    def test_scifar100(self, override_args=None):
        """Split CIFAR-100 benchmark"""
        args = create_default_args({'cuda': 0,
                                    'n_inner_updates': 5,
                                    'second_order': True,
                                    'grad_clip_norm': 1.0,
                                    'learn_lr': True,
                                    'lr_alpha': 0.25,
                                    'sync_update': False,
                                    'mem_size': 200,
                                    'lr': 0.1,
                                    'train_mb_size': 10,
                                    'train_epochs': 10,
                                    'seed': 0}, override_args)
        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                                 args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.SplitCIFAR100(n_experiences=20,
                                                 return_task_id=True)

        model = MTConvCIFAR()
        interactive_logger = avl.logging.InteractiveLogger()
        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        # LaMAML strategy
        rs_buffer = ReservoirSamplingBuffer(max_size=args.mem_size)
        replay_plugin = ReplayPlugin(
            mem_size=args.mem_size,
            batch_size=args.train_mb_size,
            batch_size_mem=args.train_mb_size,
            task_balanced_dataloader=False,
            storage_policy=rs_buffer
        )

        cl_strategy = avl.training.LaMAML(
            model,
            torch.optim.SGD(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),
            n_inner_updates=args.n_inner_updates,
            second_order=args.second_order,
            grad_clip_norm=args.grad_clip_norm,
            learn_lr=args.learn_lr,
            lr_alpha=args.lr_alpha,
            sync_update=args.sync_update,
            train_mb_size=args.train_mb_size,
            train_epochs=args.train_epochs,
            eval_mb_size=100,
            device=device,
            plugins=[replay_plugin],
            evaluator=evaluation_plugin,
        )

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"LaMAML-CIFAR100 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lamaml', 'cifar100'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_stinyimagenet(self, override_args=None):
        """Split TinyImageNet benchmark"""
        args = create_default_args({'cuda': 0,
                                    'n_inner_updates': 5,
                                    'second_order': True,
                                    'grad_clip_norm': 1.0,
                                    'learn_lr': True,
                                    'lr_alpha': 0.4,
                                    'sync_update': False,
                                    'mem_size': 400,
                                    'lr': 0.1,
                                    'train_mb_size': 10,
                                    'train_epochs': 10,
                                    'seed': 0}, override_args)
        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                                 args.cuda >= 0 else "cpu")

        benchmark = avl.benchmarks.SplitTinyImageNet(n_experiences=40,
                                                     return_task_id=True)

        model = MTConvTinyImageNet()
        interactive_logger = avl.logging.InteractiveLogger()
        evaluation_plugin = avl.training.plugins.EvaluationPlugin(
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            loggers=[interactive_logger], benchmark=benchmark)

        # LaMAML strategy
        rs_buffer = ReservoirSamplingBuffer(max_size=args.mem_size)
        replay_plugin = ReplayPlugin(
            mem_size=args.mem_size,
            batch_size=args.train_mb_size,
            batch_size_mem=args.train_mb_size,
            task_balanced_dataloader=False,
            storage_policy=rs_buffer
        )

        cl_strategy = avl.training.LaMAML(
            model,
            torch.optim.SGD(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),
            n_inner_updates=args.n_inner_updates,
            second_order=args.second_order,
            grad_clip_norm=args.grad_clip_norm,
            learn_lr=args.learn_lr,
            lr_alpha=args.lr_alpha,
            sync_update=args.sync_update,
            train_mb_size=args.train_mb_size,
            train_epochs=args.train_epochs,
            eval_mb_size=100,
            device=device,
            plugins=[replay_plugin],
            evaluator=evaluation_plugin,
        )

        for experience in benchmark.train_stream:
            cl_strategy.train(experience)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"LaMAML-TinyImageNet Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lamaml', 'cifar100'))
        if args.check and target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
