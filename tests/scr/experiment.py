import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_cifar10 import online_scr_scifar10


class SCR(unittest.TestCase):
    """
    Reproducing Supervised Contrastive Replay paper
    "Supervised Contrastive Replay: Revisiting the Nearest Class Mean Classifier
    in Online Class-Incremental Continual Learning" by Mai et. al. (2021).
    https://arxiv.org/abs/2103.13885
    """

    def test_scifar10(self):
        """Split CIFAR-10 benchmark"""
        res = online_scr_scifar10({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"SCR-SCIFAR10 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('scr', 'scifar10'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)