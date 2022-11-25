import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MultiHeadReducedResNet18
from experiments.utils import set_seed, create_default_args


def gem_scifar100(override_args=None):
    """
    "Gradient Episodic Memory for Continual Learning" by Lopez-paz et. al. (2017).
    https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html
    """
    args = create_default_args({'cuda': 0, 'patterns_per_exp': 256, 'epochs': 1,
                                'mem_strength': 0.5, 'learning_rate': 0.1, 'train_mb_size': 10,
                                'seed': None}, override_args)
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
        loggers=[interactive_logger])

    cl_strategy = avl.training.GEM(
        model, SGD(model.parameters(), lr=args.learning_rate, momentum=0.9), criterion,
        patterns_per_exp=args.patterns_per_exp, memory_strength=args.mem_strength,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = gem_scifar100()
    print(res)
