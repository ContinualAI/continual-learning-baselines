import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import gdumb_smnist


class GDumb(unittest.TestCase):
    """
    Reproducing GDumb experiments from paper
    "GDumb: A Simple Approach that Questions Our Progress in Continual Learning" by Prabhu et. al. (2020).
    https://link.springer.com/chapter/10.1007/978-3-030-58536-5_31
    """

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = gdumb_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"GDumb-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('gdumb', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
