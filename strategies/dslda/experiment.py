import unittest
import warnings
import torch
import avalanche as avl
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from torchvision import transforms
from strategies.utils import create_default_args, get_average_metric, get_target_result


class DSLDA(unittest.TestCase):
    """
    Reproducing Streaming Deep LDA experiments from the paper
    "Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis"
    by Hayes et. al. (2020).
    https://arxiv.org/abs/1909.01520
    """
    def test_core50(self, override_args=None):
        """CORe50 New Classes benchmark"""
        args = create_default_args({'cuda': 0, 'feature_size': 512, 'batch_size': 512,
                                    'shrinkage': 1e-4, 'plastic_cov': True}, override_args)
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

        for i, exp in enumerate(benchmark.train_stream):
            cl_strategy.train(exp)
            res = cl_strategy.eval(benchmark.test_stream)

        avg_stream_acc = get_average_metric(res)
        print(f"SLDA-CORe50 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('dslda', 'core50'))
        # if args.check:
        #     self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)
