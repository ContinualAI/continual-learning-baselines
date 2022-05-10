import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MultiHeadReducedResNet18
from experiments.utils import set_seed, create_default_args


def agem_scifar100(override_args=None):
    args = create_default_args({'cuda': 0, 'patterns_per_exp': 65, 'epochs': 1,
                                'sample_size': 1300, 'learning_rate': 0.03, 'train_mb_size': 10,
                                'seed': 0}, override_args)
    set_seed(args.seed)
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

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)
    return res
