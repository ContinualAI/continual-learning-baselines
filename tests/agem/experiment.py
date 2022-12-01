import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.permuted_mnist import agem_pmnist
from experiments.split_cifar100 import agem_scifar100


class AGEM(unittest.TestCase):
    """
    Reproducing Average-GEM experiments from paper
    "Efficient Lifelong Learning with A-GEM" by Chaudhry et. al. (2019).
    https://openreview.net/pdf?id=Hkf2_sC5FX
    The main difference with the original paper is that we do not append any task descriptor
    to the model input.
    We train on the last 17 experiences since we apply the evaluation protocol defined
    in the paper but we do not perform model selection.
    """
    def test_pmnist(self):
        """Permuted MNIST benchmark"""
        res = agem_pmnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"AGEM-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('agem', 'pmnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_scifar100(self):
        """Split CIFAR-100 benchmark"""
        res = agem_scifar100({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"AGEM-SCIFAR100 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('agem', 'scifar100'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.04)
