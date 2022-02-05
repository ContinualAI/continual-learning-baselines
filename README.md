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

## Coverage
The table below describes all the experiments currently implemented.

The `Command` column refers to the console command needed to reproduce experiments
from that strategy and benchmark. Please, remember to prepend `strategies.` to the command when executing it
from the project root directory.


Strategy | Benchmark           | Command (`strategies.`)          | Metrics 
---|---------------------|----------------------------------|---
Synaptic Intelligence | Split MNIST         | SynapticIntelligence.test_smnist | Stream Accuracy
Synaptic Intelligence | Permuted MNIST      | SynapticIntelligence.test_pmnist | Stream Accuracy
CoPE | Split MNIST         | COPE.test_smnist                 | Stream Accuracy
Deep Streaming LDA | CORe50              | DSLDA.test_core50                | Stream Accuracy
Elastic Weight Consolidation | Permuted MNIST      | EWC.test_pmnist                  | Stream Accuracy
Average GEM | Permuted MNIST      | AGEM.test_pmnist                 | Stream Accuracy
Average GEM | Split CIFAR 100     | AGEM.test_scifar100              | Stream Accuracy
GEM | Permuted MNIST      | GEM.test_pmnist                  | Stream Accuracy
GEM | Split CIFAR 100     | GEM.test_scifar100               | Stream Accuracy
GSS-greedy | Split MNIST         | GSS.test_smnist                  | Stream Accuracy
LwF | Split MNIST         | LwF.test_smnist                  | Stream Accuracy
LwF | Permuted MNIST      | LwF.test_pmnist                  | Stream Accuracy
LwF | Split Tiny ImageNet | LwF.test_stinyimagenet           | Stream Accuracy
GSS | Split MNIST         | GSS.test_smnist                  | Stream Accuracy
iCaRL | Split CIFAR 100     | iCARL.test_scifar100             | Average Incremental Accuracy
GDumb | Split MNIST         | GDumb.test_smnist                | Stream Accuracy

The file `strategies/target_results.csv` lists the target metric value(s) for each experiment.

## Python dependencies for experiments
Outside Python standard library, the main packages required to run the experiments are PyTorch, Avalanche and Pandas. 
* **Avalanche**: `pip install avalanche-lib` (stable version). Alternatively, you can install the latest version of Avalanche directly from the master branch of github with `pip install git+https://github.com/ContinualAI/avalanche.git`. For more details, please check out the complete guide on how to install Avalanche [here](https://avalanche.continualai.org/getting-started/how-to-install). 
* **PyTorch**: we recommend to follow [the official guide](https://pytorch.org/get-started/locally/).
* **Pandas**: `pip install pandas`. [Official guide](https://pandas.pydata.org/docs/getting_started/install.html#installing-pandas).


## Run custom experiments with Python (recommended)
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

## Command line experiments
You can run experiments directly through console with the default parameters.  
Open the console and go to the project `reproducible-continual-learning` root folder.

Execute from console 
```bash
python -m unittest strategies.{strategy_class_name}.test_{benchmark}
```
to run experiment with default parameters.  
For example
```bash
python -m unittest strategies.SynapticIntelligence.test_smnist
```
runs Synaptic Intelligence on Split MNIST.

To execute experiment with custom parameters, please refer to the previous section.

## Contribute
We are always looking for new contributors willing to help us in the challenging mission of providing robust experiments
to the community. Would you like to join us? The steps are easy!

1. Take a look at the opened issues and find yours
2. Fork this repo and write an experiment (see next section)
3. Submit a PR and receive support from the maintainers
4. Merge the PR, your contribution is now included in the project!

### Write an experiment
1. Create a folder with appropriate name (e.g., strategy name)
2. Fill the `experiment.py` file with your code (one method per benchmark) and place it under the newly created directory
3. Make the main class of your experiment visible from outside the folder (in `__init__.py` within your folder add `from .experiment import YourClassName`). Check if other `__init__` files need similar modifications.
4. Add to `target_results.csv` the expected result for your experiment. You can add a number or a list of numbers.
5. Update table in `README.md`.

Check out one of the existing strategies to better understand the logic of the experiments.
