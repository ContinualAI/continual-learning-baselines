from avalanche.training import SCR
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18, SCRModel
from avalanche.training.plugins import EvaluationPlugin
import kornia.augmentation as K

from experiments.utils import create_default_args, set_seed
from avalanche.benchmarks.scenarios import split_online_stream
from torch.utils.data import DataLoader
from avalanche.training.losses import SCRLoss


def online_scr_scifar10(override_args=None):
    """
    Reproducing Supervised Contrastive Replay paper
    "Supervised Contrastive Replay: Revisiting the Nearest Class Mean Classifier
    in Online Class-Incremental Continual Learning" by Mai et. al. (2021).
    https://arxiv.org/abs/2103.13885

    In the original paper, SCR uses the ReviewTrick
    technique (fine-tuning on the buffer at the end of training on each experience).
    For fairness of comparison with the other strategies, we do not employ
    the review trick, therefore our results
    are lower wrt the original paper. However, you can activate the review trick
    by setting the corresponding parameter to True in the args.
    """
    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 200,
            "lr": 0.1,
            "train_mb_size": 10,
            "seed": None,
            "batch_size_mem": 100,
            "review_trick": False
        },
        override_args
    )

    set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    scenario = SplitCIFAR10(
        5,
        return_task_id=False,
        train_transform=data_transform,
        eval_transform=data_transform,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
    )

    # SlimResNet18 is used as encoder
    # the projection network takes as input the output of the ResNet
    nf = 20
    encoding_network = SlimResNet18(nclasses=10, nf=nf)
    encoding_network.linear = torch.nn.Identity()
    projection_network = torch.nn.Sequential(
        torch.nn.Linear(nf*8, nf*8), torch.nn.ReLU(inplace=True), torch.nn.Linear(nf*8, 128))

    # a NCM Classifier is used at eval time
    model = SCRModel(
        feature_extractor=encoding_network,
        projection=projection_network)

    optimizer = SGD(model.parameters(), lr=args.lr)

    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]
    training_metrics = []

    # training accuracy cannot be directly monitored with SCR
    # training loss and eval loss are two different ones
    evaluation_metrics = [
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
    ]
    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )

    scr_transforms = torch.nn.Sequential(
        K.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
        K.RandomHorizontalFlip(),
        K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        K.RandomGrayscale(p=0.2)
    )
    # should achieve around 48% final accuracy
    cl_strategy = SCR(
        model=model,
        optimizer=optimizer,
        augmentations=scr_transforms,
        plugins=None,
        evaluator=evaluator,
        device=device,
        train_mb_size=args.train_mb_size,
        eval_mb_size=64,
        mem_size=args.mem_size,
        batch_size_mem=args.batch_size_mem
    )

    ocl_scenario = split_online_stream(
        original_stream=scenario.train_stream,
        experience_size=args.train_mb_size,
        access_task_boundaries=False,
    )

    for t, experience in enumerate(ocl_scenario):
        cl_strategy.train(experience)

        if args.review_trick and experience.is_last_subexp:  # at the end of each macro experience
            buffer = cl_strategy.replay_plugin.storage_policy.buffer
            dl = DataLoader(buffer, batch_size=args.batch_size_mem, shuffle=True, drop_last=True)
            model.train()
            crit = SCRLoss(temperature=0.1)
            for x, y, _ in dl:
                assert x.size(0) % 2 == 0, f"{x.size(0)}"
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                mb_x_augmented = scr_transforms(x)
                x = torch.cat([x, mb_x_augmented], dim=0)
                assert x.size(0) % 2 == 0, f"{x.size(0)}"

                out = model(x)
                assert out.size(0) % 2 == 0, f"{x.size(0)}"

                original_batch_size = int(out.size(0) / 2)
                original_examples = out[:original_batch_size]
                augmented_examples = out[original_batch_size:]
                out = torch.stack(
                    [original_examples, augmented_examples],
                    dim=1)

                loss = crit(out, y)
                loss.backward()
                params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                grad = [p.grad.clone()/10. for p in params]
                for g, p in zip(grad, params):
                    p.grad.data.copy_(g)
                optimizer.step()
                model.eval()
                cl_strategy.compute_class_means()

    if args.review_trick:
        model.eval()
        cl_strategy.compute_class_means()

    results = cl_strategy.eval(scenario.test_stream)
    return results


if __name__ == '__main__':
    res = online_scr_scifar10()
    print(res)
