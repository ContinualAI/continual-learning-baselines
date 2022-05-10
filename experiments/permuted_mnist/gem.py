import avalanche as avl
import torch
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MLP
from experiments.utils import set_seed, create_default_args


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


def gem_pmnist(override_args=None):
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

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = gem_pmnist()
    print(res)
