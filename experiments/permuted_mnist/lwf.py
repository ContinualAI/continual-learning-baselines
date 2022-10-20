import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MLP
from experiments.utils import set_seed, create_default_args


class LwFCEPenalty(avl.training.LwF):
    """This wrapper around LwF computes the total loss
    by diminishing the cross-entropy contribution over time,
    as per the paper
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf
    The loss is L_tot = (1/n_exp_so_far) * L_cross_entropy +
                        alpha[current_exp] * L_distillation
    """
    def _before_backward(self, **kwargs):
        self.loss *= float(1/(self.clock.train_exp_counter+1))
        super()._before_backward(**kwargs)


def lwf_pmnist(override_args=None):
    """"
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this experiment is based on
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf

    Please, note that the performance of LwF on Permuted MNIST is below the one achieved
    by Naive with the same configuration. This is compatible with the results presented
    by van de Ven et. al. (2018).
    """
    args = create_default_args({'cuda': 0, 'lwf_alpha': [0.]+[1-(1./float(i)) for i in range(2, 11)],
                                'lwf_temperature': 2, 'epochs': 11,
                                'layers': 2, 'hidden_size': 300,
                                'learning_rate': 0.0001, 'train_mb_size': 128, 'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.PermutedMNIST(10)
    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.layers)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = LwFCEPenalty(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = lwf_pmnist()
    print(res)
