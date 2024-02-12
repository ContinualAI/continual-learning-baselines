import torch.nn as nn

from avalanche.benchmarks import CLExperience
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks import data_incremental_benchmark
from avalanche.evaluation.metrics import \
    accuracy_metrics, \
    loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import GSS_greedy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from experiments.utils import set_seed, create_default_args
from models import MLP_gss


def gss_smnist(override_args=None):
    """
    https://arxiv.org/abs/1903.08671

    Expected accuracy is 82% which is slightly higher than the one we achieve.
    """
    args = create_default_args({
        'cuda': 0, 'lr': 0.05,
        'train_mb_size': 10, 'mem_strength': 10,
        'input_size': [1, 28, 28], 'train_epochs': 3, 'eval_mb_size': 10,
        'mem_size': 300, 'seed': None}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    model, benchmark = setup_mnist()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(stream=True), loggers=[InteractiveLogger()])

    optimizer = SGD(model.parameters(), lr=args.lr)
    strategy = GSS_greedy(model, optimizer, criterion=CrossEntropyLoss(),
                          mem_strength=args.mem_strength,
                          input_size=args.input_size,
                          train_epochs=args.train_epochs,
                          train_mb_size=args.train_mb_size,
                          eval_mb_size=args.eval_mb_size,
                          mem_size=args.mem_size,
                          device=device,
                          evaluator=eval_plugin)

    res = None
    for experience in benchmark.train_stream:
        print(">Experience ", experience.current_experience)
        strategy.train(experience)
        res = strategy.eval(benchmark.test_stream)

    return res


def shrinking_experience_size_split_strategy(
        experience: CLExperience):

    experience_size = 1000

    exp_dataset = experience.dataset
    exp_indices = list(range(len(exp_dataset)))

    result_datasets = []

    exp_indices = \
        torch.as_tensor(exp_indices)[
            torch.randperm(len(exp_indices))
        ].tolist()

    result_datasets.append(exp_dataset.subset(exp_indices[0:experience_size]))

    return result_datasets


def setup_mnist():

    scenario = data_incremental_benchmark(SplitMNIST(
        n_experiences=5, seed=1), experience_size=0,
        custom_split_strategy=shrinking_experience_size_split_strategy)
    n_inputs = 784
    nh = 100
    nl = 2
    n_outputs = 10
    model = MLP_gss([n_inputs] + [nh] * nl + [n_outputs])

    return model, scenario


if __name__ == '__main__':
    res = gss_smnist()
    print(res)
