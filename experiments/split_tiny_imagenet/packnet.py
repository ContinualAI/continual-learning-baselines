"""
Reproduce Delange et al. (2021) benchmark results for PackNet
(Mallya & Lazebnik, 2018) on Split Tiny ImageNet

Delange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., 
  Slabaugh, G., & Tuytelaars, T. (2021). A continual learning survey: Defying 
  forgetting in classification tasks. IEEE Transactions on Pattern Analysis 
  and Machine Intelligence, 1–1. https://doi.org/10.1109/TPAMI.2021.3057446

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single
  Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer Vision
  and Pattern Recognition, 7765–7773. https://doi.org/10.1109/CVPR.2018.00810
"""

import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from experiments.utils import set_seed, create_default_args
from models.vgg import SingleHeadVGGSmall
from avalanche.models.packnet import PackNetModel, packnet_simple_mlp
from avalanche.training.supervised.strategy_wrappers import PackNet


def packnet_stinyimagenet(override_args=None):
    """
    The original PackNet paper uses an unusual experimental setup, so we
    base this experiment on Delange et al. (2021) benchmark.

    Delange et al. (2021) set prune_proportion using the Continual
    Hyperparameter Selection Framework. We instead use `prune_proportion` such
    that the number of parameters in each task-specific-subset of the model
    are roughly equal.
    """
    args = create_default_args(
        {
            "cuda": 0,
            "epochs": 30,
            "learning_rate": 1e-3,
            "train_mb_size": 200,
            "seed": 42,
            "dataset_root": None,
            "prune_proportion": [
                0.90,
                0.88,
                0.87,
                0.85,
                0.83,
                0.80,
                0.75,
                0.66,
                0.50,
                0.00,
            ],
            "post_prune_epochs": 15,
        },
        override_args,
    )
    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    benchmark = avl.benchmarks.SplitTinyImageNet(
        10, return_task_id=True, dataset_root=args.dataset_root
    )
    model = SingleHeadVGGSmall(n_classes=200)
    model = PackNetModel(model)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    cl_strategy = PackNet(
        model,
        Adam(model.parameters(), lr=args.learning_rate),
        post_prune_epochs=args.post_prune_epochs,
        prune_proportion=args.prune_proportion,
        criterion=criterion,
        train_mb_size=args.train_mb_size,
        train_epochs=args.epochs,
        eval_mb_size=128,
        device=device,
        evaluator=evaluation_plugin,
    )

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = packnet_stinyimagenet()
    print(res)
