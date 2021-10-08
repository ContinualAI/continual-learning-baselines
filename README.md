<div align="center">
    
# Reproducible Continual Learning
**[Avalanche Website](https://avalanche.continualai.org)** | **[Avalanche Repository](https://github.com/ContinualAI/avalanche)**

</div>

<p align="center">
    <img src="https://www.dropbox.com/s/90thp7at72sh9tj/avalanche_logo_with_clai.png?raw=1"/>
</p>



**The aim of this project is to provide to the continual learning community a set of experiments validating and
reproducing existing works in continual learning.**

You can use these experiments to better understand the specific configuration needed to achieve the results of a paper,
or maybe to play around with different hyper-parameters and such. Sky is the limit!

To guarantee fair implementations, we rely on the **[Avalanche](https://github.com/ContinualAI/avalanche)** library, developed and maintained by *[ContinualAI](https://www.continualai.org/)*.
Feel free to check it out and support the project!


## How to run experiments with Python
Experiments can be run with a python script by simply:
1. Creating an instance of the strategy object
2. Executing the strategy on a benchmark by running the related method of the strategy object
3. Look at console output for details of ongoing experiment

```python
from strategies import SynapticIntelligence  # select the strategy

s = SynapticIntelligence()  # create the strategy

# run the experiment with custom parameters and without performing `assert` checks
s.test_smnist({'learning_rate': 1e-3, 'si_lambda': 1,
               'check': False})
```

## How to run experiments from command line
Open console and go to the project `reproducible-continual-learning` root folder.

Execute from console `python -m unittest strategies.{strategy_class_name}.test_{benchmark}` to run experiment with default parameters.  
For example, `python -m unittest strategies.SynapticIntelligence.test_smnist` runs Synaptic Intelligence on Split MNIST.

To execute experiment with custom parameters, please refer to the previous section.

## How to contribute
We are always looking for new contributors willing to help us in the challenging mission of providing robust experiments
to the community. Would you like to join us? The steps are easy!

1. Take a look at the opened issues and find yours
2. Fork this repo and make the changes (see next section)
3. Submit a PR and receive support from the maintainers
4. Merge the PR, your contribution is now included in the project!

## How to write an experiment
1. Create a folder with appropriate name (e.g., strategy name)
2. Fill the `experiment.py` file with your code (one method per benchmark) and place it under the newly created directory
3. Make the main class of your experiment visible from outside the folder (in `__init__.py` within your folder add `from .experiment import YourClassName`)
4. Add to `target_results.csv` the expected result for your experiment. You can add a number or a list of numbers.

Check out the synaptic intelligence example to better understand the logic of experiments.