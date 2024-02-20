import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.permuted_mnist import gem_pmnist
from experiments.split_cifar100 import gem_scifar100


class GEM(unittest.TestCase):
    """
    Reproducing GEM experiments from paper
    "Gradient Episodic Memory for Continual Learning" by Lopez-paz et. al. (2017).
    https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html
    """

    def test_pmnist(self):
        """Permuted MNIST benchmark"""
        res = gem_pmnist({'seed': 0, 'n_exp': 5})
        avg_stream_acc = get_average_metric(res)
        print(f"GEM-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('gem', 'pmnist'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_scifar100(self):
        """Split CIFAR-100 benchmark"""
        res = gem_scifar100({'seed': 435342})
        avg_stream_acc = get_average_metric(res)
        print(f"GEM-SCIFAR100 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('gem', 'scifar100'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
