import unittest
import torch
import avalanche as avl
from tests.utils import get_average_metric, get_target_result


@unittest.skip("Just a template, skipping this test.")  # remove this when implementing the test
class StrategyName(unittest.TestCase):
    def test_benchmarkname(self):
        pass

        #####################
        # FILL HERE         #
        #####################
        # Create your experiment with Avalanche and put it in the `experiments` folder
        # in the project root directory
        # get the final results into the res variable
        # res = mystrategy_benchmark(args)

        #####################
        # Process results   #
        #####################
        # you may find useful the already imported functions
        # `get_average_metric` and `get_target_result`

        # example:
        # acc = get_average_metric(res)
        # target_acc = float(get_target_result('strategy', 'benchmark'))

        # check that your current result meets the expected result

        # example:
        # if target_acc > avg_stream_acc:
        #   self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)
