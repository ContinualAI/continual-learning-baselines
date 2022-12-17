import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import rwalk_smnist


class RWalk(unittest.TestCase):
    """
    Reproducing RWalk experiments from paper
    "Riemannian Walk for Incremental Learning:
    Understanding Forgetting and Intransigence" by Chaudhry et. al. (2018).
    https://openaccess.thecvf.com/content_ECCV_2018/html/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.html
    """

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = rwalk_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"RWALK-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('rwalk', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.01)