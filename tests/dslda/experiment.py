import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.core50 import deep_slda_core50


class DSLDA(unittest.TestCase):
    """
    Reproducing Streaming Deep LDA experiments from the paper
    "Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis"
    by Hayes et. al. (2020).
    https://arxiv.org/abs/1909.01520
    """
    def test_core50(self):
        """CORe50 New Classes benchmark"""
        res = deep_slda_core50({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"DSLDA-CORe50 Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('dslda', 'core50'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
