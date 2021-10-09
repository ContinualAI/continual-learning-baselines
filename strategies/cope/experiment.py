import unittest
import torch
import avalanche as avl
from strategies.utils import create_default_args, get_average_metric, get_target_result
from models import MLP


class COPE(unittest.TestCase):
    """
    Reproducing CoPE experiments from the paper
    "Continual prototype evolution: Learning online from non-stationary data streams"
    by De Lange et. al. (2021).
    https://arxiv.org/abs/2009.00919
    """
    def test_smnist(self, override_args=None):
        """Split MNIST benchmark"""
        args = create_default_args({'cuda': 0, 'nb_tasks': 5, 'batch_size': 10, 'epochs': 1,
                                    'mem_size': 2000, 'alpha': 0.99, 'featsize': 32}, override_args)
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        n_classes = 10
        task_scenario = avl.benchmarks.SplitMNIST(args.nb_tasks, return_task_id=False,
                                                  fixed_class_order=[i for i in range(n_classes)])

        # Make data incremental (one batch = one experience)
        benchmark = avl.benchmarks.data_incremental_benchmark(task_scenario,
                                                              experience_size=args.batch_size)
        print(f"{benchmark.n_experiences} batches in online data incremental setup.")
        # 6002 batches for SplitMNIST with batch size 10
        # ---------

        model = MLP(output_size=args.featsize,
                    hidden_size=400, hidden_layers=2, drop_rate=0)

        logger = avl.logging.InteractiveLogger()

        eval_plugin = avl.training.plugins.EvaluationPlugin(
            avl.evaluation.metrics.accuracy_metrics(experience=True, stream=True),
            avl.evaluation.metrics.loss_metrics(experience=False, stream=True),
            avl.evaluation.metrics.StreamForgetting(),
            loggers=[logger],
            benchmark=benchmark)

        cope = avl.training.plugins.CoPEPlugin(mem_size=args.mem_size, alpha=args.alpha,
                                               p_size=args.featsize, n_classes=n_classes)

        cl_strategy = avl.training.Naive(model, torch.optim.SGD(model.parameters(), lr=0.01),
                            cope.ppp_loss,  # CoPE PPP-Loss
                            train_mb_size=args.batch_size, train_epochs=args.epochs,
                            eval_mb_size=100, device=device,
                            plugins=[cope],
                            evaluator=eval_plugin
                            )

        cl_strategy.train(benchmark.train_stream)
        res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"COPE-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('cope', 'smnist'))
        if args.check:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)
