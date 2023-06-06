import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.permuted_mnist import mir_pmnist
from experiments.split_mnist import mir_smnist
from experiments.split_cifar10 import mir_scifar10


class MIR(unittest.TestCase):
    """
    Reproducing MIR experiments from paper
    "Online Continual Learning With Maximally Interfered Retrieval" by R. Aljundi et. al. (2019)
    https://papers.nips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-MetaReview.html
    """

    def test_pmnist(self):
        """Permuted MNIST benchmark"""
        res = mir_pmnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"MIR-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('mir', 'pmnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = mir_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"MIR-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('mir', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_scifar10(self):
        """Split CIFAR-10 benchmark"""
        res = mir_scifar10({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"MIR-SCIFAR10 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('mir', 'scifar10'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
