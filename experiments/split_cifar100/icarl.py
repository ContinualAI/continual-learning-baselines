import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import SGD
from torchvision import transforms

from experiments.utils import set_seed, create_default_args

from avalanche.benchmarks import SplitCIFAR100
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import *
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training import ICaRL


def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode='constant')
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+32), crop[1]:(crop[1]+32)]
    else:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+32), crop[1]:(crop[1]+32)][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t


def icarl_scifar100(override_args=None):
    """
    "iCaRL: Incremental Classifier and Representation Learning",
    Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert;
    Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2001-2010
    https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html

    The expected performance is 50%, while we achieve a lower score.
    """
    args = create_default_args({'cuda': 0, 'batch_size': 128, 'nb_exp': 10,
                                'memory_size': 2000, 'epochs': 70, 'lr_base': 2.,
                                'lr_milestones': [49, 63], 'lr_factor': 5.,
                                'wght_decay': 0.00001, 'seed': None}, override_args)

    # class incremental learning: classes mutual exclusive
    fixed_class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
                         94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
                         84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
                         69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
                         17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
                         1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
                         38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
                         60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
                         40, 30, 23, 85, 2, 95, 56, 48, 71, 64,
                         98, 13, 99, 7, 34, 55, 54, 26, 35, 39]
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")

    transform_prototypes = transforms.Compose([
        icarl_cifar100_augment_data,
    ])

    train_trsf = transforms.Compose([transforms.ToTensor(),
                                     icarl_cifar100_augment_data])
    test_trsf = transforms.Compose([transforms.ToTensor()])

    benchmark = SplitCIFAR100(
        n_experiences=args.nb_exp, seed=args.seed,
        fixed_class_order=fixed_class_order,
        train_transform=train_trsf,
        eval_transform=test_trsf
    )

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loggers=[interactive_logger])

    # _____________________________Strategy
    model: IcarlNet = make_icarl_net(num_classes=100)
    model.apply(initialize_icarl_net)

    optim = SGD(model.parameters(), lr=args.lr_base,
                weight_decay=args.wght_decay, momentum=0.9)
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, args.lr_milestones, gamma=1.0 / args.lr_factor))

    strategy = ICaRL(
        model.feature_extractor, model.classifier, optim,
        args.memory_size,
        buffer_transform=transform_prototypes,
        fixed_memory=True, train_mb_size=args.batch_size,
        train_epochs=args.epochs, eval_mb_size=args.batch_size,
        plugins=[sched], device=device, evaluator=eval_plugin
    )

    for i, exp in enumerate(benchmark.train_stream):
        strategy.train(exp, num_workers=4)
        res = strategy.eval(benchmark.test_stream, num_workers=4)

    return res


if __name__ == '__main__':
    res = icarl_scifar100()
    print(res)
