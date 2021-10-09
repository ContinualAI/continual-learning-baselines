import unittest
import torch
import avalanche as avl
from strategies.utils import create_default_args, get_average_metric, get_target_result


class StrategyName(unittest.TestCase):
    def test_benchmarkname(self, override_args=None):
        #####################
        # Create arguments  #
        #####################
        # add as many parameters as you want in the input dictionary
        args = create_default_args({'cuda': 0}, override_args)
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        #####################
        # FILL HERE         #
        #####################
        # Create your experiment with Avalanche and get the final results into the res variable
        # (e.g. res = strategy.eval(benchmark.test_stream)

        #####################
        # Process results   #
        #####################
        # you may find useful the already imported functions
        # `get_average_metric` and `get_target_result`
        if args.check:
            pass
            # check that your current result meets the expected result
            # e.g., self.assertAlmostEqual(target_acc, expected_acc, delta=0.02)
