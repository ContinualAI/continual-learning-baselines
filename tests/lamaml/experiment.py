import unittest
from tests.utils import get_target_result, get_average_metric
from experiments.split_cifar100 import lamaml_scifar100
from experiments.split_tiny_imagenet import lamaml_stinyimagenet


class LaMAML(unittest.TestCase):
    """
        Reproducing LaMAML experiments from paper
        "La-MAML: Look-ahead Meta Learning for Continual Learning",
        Gunshi Gupta, Karmesh Yadav, Liam Paull;
        NeurIPS, 2020
        https://arxiv.org/abs/2007.13904
    """

    def test_scifar100(self):
        """
            scifar100, multi-pass
        """
        res = lamaml_scifar100({'seed': 498235})
        avg_stream_acc = get_average_metric(res)
        print(f"LaMAML-SCIFAR100 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lamaml', 'scifar100'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)

    def test_stinyimagenet(self):
        """
            stinyimagenet, multi-pass
        """
        res = lamaml_stinyimagenet({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"LaMAML-SplitTinyImageNet Average Stream Accuracy: " + \
              f"{avg_stream_acc:.2f}")

        target_acc = float(get_target_result('lamaml', 'stiny-imagenet'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
