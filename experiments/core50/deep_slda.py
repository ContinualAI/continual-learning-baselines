import warnings
import torch
import avalanche as avl
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from torchvision import transforms
from experiments.utils import set_seed, create_default_args


def deep_slda_core50(override_args=None):
    """
    "Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis"
    by Hayes et. al. (2020).
    https://arxiv.org/abs/1909.01520
    """
    args = create_default_args({'cuda': 0, 'feature_size': 512, 'batch_size': 512,
                                'shrinkage': 1e-4, 'plastic_cov': True, 'seed': None}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    _mu = [0.485, 0.456, 0.406]  # imagenet normalization
    _std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mu,
                             std=_std)
    ])

    benchmark = avl.benchmarks.CORe50(scenario='nc', train_transform=transform, eval_transform=transform)

    eval_plugin = avl.training.plugins.EvaluationPlugin(
        loss_metrics(epoch=True, experience=True, stream=True),
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )

    criterion = torch.nn.CrossEntropyLoss()
    model = avl.models.SLDAResNetModel(device=device, arch='resnet18',
                                       imagenet_pretrained=True)

    cl_strategy = avl.training.StreamingLDA(model, criterion,
                                            args.feature_size, num_classes=50,
                                            eval_mb_size=args.batch_size,
                                            train_mb_size=args.batch_size,
                                            train_epochs=1,
                                            shrinkage_param=args.shrinkage,
                                            streaming_update_sigma=args.plastic_cov,
                                            device=device, evaluator=eval_plugin)

    warnings.warn(
        "The Deep SLDA example is not perfectly aligned with "
        "the paper implementation since it does not use a base "
        "initialization phase and instead starts streming from "
        "pre-trained weights. Performance should still match.")

    res = None
    for i, exp in enumerate(benchmark.train_stream):
        cl_strategy.train(exp)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = deep_slda_core50()
    print(res)
