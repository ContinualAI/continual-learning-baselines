import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.permuted_mnist import ewc_pmnist


class EWC(unittest.TestCase):
    """
    Reproducing Elastic Weight Consolidation experiments from paper
    "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et. al. (2017).
    https://www.pnas.org/content/114/13/3521
    """

    def test_pmnist(self):
        """Permuted MNIST benchmark"""
        res = ewc_pmnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"EWC-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('ewc', 'pmnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
