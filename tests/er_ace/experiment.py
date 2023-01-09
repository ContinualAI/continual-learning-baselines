import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_cifar10 import er_ace


class ER_ACE(unittest.TestCase):
    """
    Reproducing MIR experiments from paper
    "Online Continual Learning With Maximally Interfered Retrieval" by R. Aljundi et. al. (2019)
    https://papers.nips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-MetaReview.html
    """

    def test_scifar10(self):
        """Split CIFAR-10 benchmark"""
        res = er_ace_scifar10()
        avg_stream_acc = get_average_metric(res)
        print(f"ER_ACE-SCIFAR10 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('er_ace', 'scifar10'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
