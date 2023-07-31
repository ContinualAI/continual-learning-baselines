import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_cifar10 import eraml_scifar10
from experiments.split_cifar100 import eraml_scifar100


class ER_AML(unittest.TestCase):
    """
    Reproducing ER-AML experiments from paper
    "New insights on Reducing Abrupt Representation Change in Online Continual Learning"
    by Lucas Caccia et. al
    https://openreview.net/forum?id=N8MaByOzUfb
    """

    @unittest.skip("ER-AML is not yet in avalanche")
    def test_scifar10(self):
        """Split CIFAR-10 benchmark"""
        res = eraml_scifar10({"seed": 0})
        avg_stream_acc = get_average_metric(res)
        print(f"ER_AML-SCIFAR10 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result("er_aml", "scifar10"))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    @unittest.skip("ER-AML is not yet in avalanche")
    def test_scifar100(self):
        """Split CIFAR-100 benchmark"""
        res = eraml_scifar100({"seed": 0})
        avg_stream_acc = get_average_metric(res)
        print(f"ER_AML-SCIFAR100 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result("er_aml", "scifar100"))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
