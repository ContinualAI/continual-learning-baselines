"""
This script shows how to run an experiment on a specific strategy and benchmark.
The experiment is defined as a method of the strategy class, one method per benchmark.
You can override default parameters by providing a dictionary as input to the method.
You can find all the parameters used by the experiment in the source file of the experiment.
"""

from strategies import SynapticIntelligence  # select the strategy

s = SynapticIntelligence()  # create the strategy

# run the experiment on a specific benchmark with custom parameters
# do not provide arguments to use default parameters
s.test_smnist({'learning_rate': 1e-3, 'si_lambda': 1,
               'check': False})
# the `check` flag controls whether the `assert` command used to compare the result
# with the target performance is executed or not
