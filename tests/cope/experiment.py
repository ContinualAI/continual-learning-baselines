import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import cope_smnist


class COPE(unittest.TestCase):
    """
    Reproducing CoPE experiments from the paper
    "Continual prototype evolution: Learning online from non-stationary data streams"
    by De Lange et. al. (2021).
    https://arxiv.org/abs/2009.00919
    """

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = cope_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"COPE-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('cope', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
