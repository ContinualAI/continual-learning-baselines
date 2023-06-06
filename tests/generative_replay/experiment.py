import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import generative_replay_smnist


class GenerativeReplay(unittest.TestCase):
    """
    "Continual Learning with Deep Generative Replay" by Shin et. al. (2017).
    https://arxiv.org/abs/1705.08690
    """

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = generative_replay_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"GenerativeReplay-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('generative_replay', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
