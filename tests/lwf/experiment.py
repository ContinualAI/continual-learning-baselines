import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import lwf_smnist
from experiments.split_tiny_imagenet import lwf_stinyimagenet


class LwF(unittest.TestCase):
    """
    Reproducing Learning without Forgetting. Original paper is
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this class reproduces LwF experiments
    on Split MNIST from
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf
    We managed to surpass the performances reported in the paper by slightly
    changing the model architecture or the training hyperparameters.
    Experiments on Tiny Image Net are taken from
    "A continual learning survey: Defying forgetting in classification tasks" De Lange et. al. (2021).
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9349197
    """

    def test_smnist(self):
        """Split MNIST benchmark"""
        res = lwf_smnist({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"LwF-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lwf', 'smnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.01)

    def test_stinyimagenet(self):
        """Split Tiny ImageNet benchmark"""
        res = lwf_stinyimagenet({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"LwF-SplitTinyImageNet Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lwf', 'stiny-imagenet'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
