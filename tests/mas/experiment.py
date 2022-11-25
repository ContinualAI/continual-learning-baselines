import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_tiny_imagenet import mas_stinyimagenet


class MAS(unittest.TestCase):
    """
    Reproducing Memory Aware Synapses experiments from paper
    "A continual learning survey: Defying forgetting in classification tasks"
    by De Lange et al.
    https://doi.org/10.1109/TPAMI.2021.3057446
    """

    def test_stinyimagenet(self):
        """Split Tiny ImageNet benchmark"""
        res = mas_stinyimagenet({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print("MAS-SplitTinyImageNet Average "
              f"Stream Accuracy: {avg_stream_acc:.2f}")

        # Recover target from CSV
        target = get_target_result('mas', 'stiny-imagenet')
        if isinstance(target, list):
            target_acc = target[0]
        else:
            target_acc = target
        target_acc = float(target_acc)

        print(f"The target value was {target_acc:.2f}")

        # Check if the result is close to the target
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
