import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import synaptic_intelligence_smnist
from experiments.permuted_mnist import synaptic_intelligence_pmnist


class SynapticIntelligence(unittest.TestCase):
    """
    Reproducing Synaptic Intelligence experiments from paper
    "Continual Learning Through Synaptic Intelligence" by Zenke et. al. (2017).
    http://proceedings.mlr.press/v70/zenke17a.html
    """

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = synaptic_intelligence_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"SI-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('si', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.01)

    def test_pmnist(self):
        """Permuted MNIST benchmark"""
        res = synaptic_intelligence_pmnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"SI-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('si', 'pmnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
