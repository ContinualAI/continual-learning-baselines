import unittest
import torch
import avalanche as avl
from strategies.utils import create_default_args, get_average_stream_acc, get_target_result


class SynapticIntelligence(unittest.TestCase):
    def test_smnist(self, override_args=None):
        #####################
        # Create arguments  #
        #####################
        args = create_default_args({'cuda': 0, 'si_lambda': 1, 'si_eps': 0.001, 'epochs': 10,
                                    'learning_rate': 0.001, 'train_mb_size': 64}, override_args)
        #####################
        # Select device     #
        #####################
        device = torch.device(f"cuda:{args.cuda}"
                              if torch.cuda.is_available() and
                              args.cuda >= 0 else "cpu")

        #####################
        # FILL HERE         #
        #####################
        # Create your experiment with Avalanche and get the final results into the res variable

        #####################
        # Process results   #
        #####################
        # check the utilities `get_average_stream_acc` and `get_target_result`, they can be useful here
        if args.check:
            pass
            # check that your current result meets the expected result
            # e.g., self.assertAlmostEqual(target_acc, expected_acc, delta=0.02)
