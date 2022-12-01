import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MultiHeadMLP
from experiments.utils import set_seed, create_default_args


def rwalk_smnist(override_args=None):
    """
    Reproducing RWalk experiments from paper
    "Riemannian Walk for Incremental Learning:
    Understanding Forgetting and Intransigence" by Chaudhry et. al. (2018).
    https://openaccess.thecvf.com/content_ECCV_2018/html/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.html

    The expected value is 99%, which is higher than the achieved one.
    """
    args = create_default_args({'cuda': 0, 'ewc_lambda': 0.1, 'ewc_alpha': 0.9, 'delta_t': 10,
                                'epochs': 10, 'learning_rate': 0.001,
                                'train_mb_size': 64, 'seed': None},
                               override_args)
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
        loggers=[interactive_logger])

    cl_strategy = avl.training.Naive(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        plugins=[avl.training.plugins.RWalkPlugin(
            ewc_lambda=args.ewc_lambda,
            ewc_alpha=args.ewc_alpha,
            delta_t=args.delta_t)],
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = rwalk_smnist()
    print(res)
