import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import gss_smnist


class GSS(unittest.TestCase):
    """ GSS experiments from the original paper.

    This example the strategy GSS_greedy on Split MNIST.
    The final accuracy is around 77.96% (std 3.5)

    reference: https://arxiv.org/abs/1903.08671
    """

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = gss_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"GSS-Split MNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('gss', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
