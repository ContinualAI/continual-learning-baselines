<div align="center">
    
# Continual Learning Baselines
**[Avalanche Website](https://avalanche.continualai.org)** | **[Avalanche Repository](https://github.com/ContinualAI/avalanche)**

</div>

<p align="center">
    <img src="https://www.dropbox.com/s/90thp7at72sh9tj/avalanche_logo_with_clai.png?raw=1"/>
</p>



**This project provides a set of examples with popular continual learning strategies and baselines. 
You can easily run experiments to reproduce results from original paper or tweak the hyperparameters to get your own results.  Sky is the limit!**

To guarantee fair implementations, we rely on the **[Avalanche](https://github.com/ContinualAI/avalanche)** library, developed and maintained by *[ContinualAI](https://www.continualai.org/)*.
Feel free to check it out and support the project!

## Experiments
The tables below describes all the experiments currently implemented in the `experiments` folder, along with their result.
The tables are not meant to compare different methods but rather as a reference for their performance. Different methods may use 
slightly different setups (e.g., starting from a pre-trained model or from scratch), so it does not always make sense to compare them.

If an experiment reproduces exactly the results of a paper in terms of `Performance` (even if with different hyper-parameters), it is marked with ✅ on the `Reproduced` column. Otherwise, it is marked with ❌.   
`Avalanche` means that we could not find any specific paper as reference and we used the performance of Avalanche obtained when the strategy was first add to the library.  
If the `Performance` is much worse than the expected one, the `bug` tag is used in the `Reproduced` column.    
Finally, the `Reference` column reports the expected performance, together with a link to the associated paper (if any). Note that the link does not always point to the paper which introduced the strategy, since it sometimes differs from the one we used to get the target performance.

ACC means the Average Accuracy on all experiences after training on the last experience.   

First, we report the results for the **non-online** continual learning case (a.k.a. batch continual learning). Then, we report the results for the **online continual learning** case.

### Batch Continual Learning (non-online)

|     Benchmarks      |              Strategy               |      Scenario      | Performance | Reference                                                                                                                  | Reproduced    |
|:-------------------:|:-----------------------------------:|:------------------:|:-----------:|:---------------------------------------------------------------------------------------------------------------------------|:--------------|
|   Permuted MNIST    |    Less-Forgetful Learning (LFL)    | Domain-Incremental |  ACC=0.88   | ACC=0.88                                                                                                                   | ✅ `Avalanche` | 
|   Permuted MNIST    | Elastic Weight Consolidation (EWC)  | Domain-Incremental |  ACC=0.83   | [ACC=0.94](https://www.pnas.org/content/114/13/3521)                                                                       | ❌             |
|   Permuted MNIST    |     Synaptic Intelligence (SI)      | Domain-Incremental |  ACC=0.83   | [ACC=0.95](http://proceedings.mlr.press/v70/zenke17a.html)                                                                 | ❌             |
|   Split CIFAR-100   |               LaMAML                |  Task-Incremental  |  ACC=0.70   | [ACC=0.70](https://arxiv.org/abs/2007.13904)                                                                               | ✅             |
|   Split CIFAR-100   |                iCaRL                | Class-Incremental  |  ACC=0.48   | [ACC=0.50](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html) | ✅             |
|   Split CIFAR-100   |               Replay                | Class-Incremental  |  ACC=0.32   | ACC=0.32                                                                                                                   | ✅ `Avalanche` |
|     Split MNIST     |                RWalk                |  Task-Incremental  |  ACC=0.99   | [ACC=0.99](https://openaccess.thecvf.com/content_ECCV_2018/html/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.html)     | ✅             |
|     Split MNIST     |     Synaptic Intelligence (SI)      |  Task-Incremental  |  ACC=0.97   | [ACC=0.97](http://proceedings.mlr.press/v70/zenke17a.html)                                                                 | ✅             |
|     Split MNIST     |                GDumb                | Class-Incremental  |  ACC=0.97   | [ACC=0.97](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_31)                                                 | ✅             |
|     Split MNIST     |             GSS_greedy              | Class-Incremental  |  ACC=0.82   | [ACC=0.78](https://arxiv.org/abs/1903.08671)                                                                               | ❌             |
|     Split MNIST     |       Generative Replay (GR)        | Class-Incremental  |  ACC=0.75   | [ACC=0.75](https://arxiv.org/abs/1705.08690)                                                                               | ✅             |
|     Split MNIST     |  Learning without Forgetting (LwF)  | Class-Incremental  |  ACC=0.23   | [ACC=0.23](https://arxiv.org/pdf/1904.07734.pdf)                                                                           | ✅             |
| Split Tiny ImageNet |               LaMAML                |  Task-Incremental  |  ACC=0.54   | [ACC=0.66](https://arxiv.org/abs/2007.13904)                                                                               | ❌             |
| Split Tiny ImageNet |  Learning without Forgetting (LwF)  |  Task-Incremental  |  ACC=0.44   | [ACC=0.44](https://arxiv.org/pdf/1904.07734.pdf)                                                                           | ✅             |
| Split Tiny ImageNet |     Memory Aware Synapses (MAS)     |  Task-Incremental  |  ACC=0.40   | [ACC=0.40](https://doi.org/10.1109/TPAMI.2021.3057446)                                                                     | ✅             |
| Split Tiny ImageNet |              PackNet                |  Task-Incremental  |  ACC=0.46   | [ACC=0.47](https://doi.org/10.1109/TPAMI.2021.3057446) (Table 4 `SMALL`)                                                   | ✅             |

### Online Continual Learning

|   Benchmarks    |              Strategy               |      Scenario      | Performance | Reference                                                                                                 | Reproduced    |
|:---------------:|:-----------------------------------:|:------------------:|:-----------:|:----------------------------------------------------------------------------------------------------------|:--------------|
|     CORe50      |     Deep Streaming LDA (DSLDA)      | Class-Incremental  |  ACC=0.79   | [ACC=0.79](https://arxiv.org/abs/1909.01520)                                                              | ✅             | 
| Permuted MNIST  |                 GEM                 | Domain-Incremental |  ACC=0.80   | [ACC=0.83](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) | ✅             |
| Split CIFAR-10  |            Online Replay            | Class-Incremental  |  ACC=0.50   | ACC=0.50                                                                                                  | ✅ `Avalanche` |
| Split CIFAR-10  |               ER-AML                | Class-Incremental  |  ACC=0.47   | [ACC=0.47](https://openreview.net/forum?id=N8MaByOzUfb)                                                   | ✅             |
| Split CIFAR-10  |               ER-ACE                | Class-Incremental  |  ACC=0.45   | [ACC=0.52](https://openreview.net/forum?id=N8MaByOzUfb)                                                   | ✅             |
| Split CIFAR-10  | Supervised Contrastive Replay (SCR) | Class-Incremental  |  ACC=0.36   | [ACC=0.48](https://ieeexplore.ieee.org/document/9522763)                                                  | ✅ `Avalanche` |
| Permuted MNIST  |         Average GEM (AGEM)          | Domain-Incremental |  ACC=0.81   | [ACC=0.81](https://openreview.net/pdf?id=Hkf2_sC5FX)                                                      | ✅             | 
| Split CIFAR-100 |                 GEM                 |  Task-Incremental  |  ACC=0.63   | [ACC=0.63](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) | ✅             |
| Split CIFAR-100 |         Average GEM (AGEM)          |  Task-Incremental  |  ACC=0.62   | [ACC=0.62](https://openreview.net/pdf?id=Hkf2_sC5FX)                                                      | ✅             |
| Split CIFAR-100 |               ER-ACE                | Class-Incremental  |  ACC=0.24   | [ACC=0.25](https://openreview.net/forum?id=N8MaByOzUfb)                                                   | ✅             |
| Split CIFAR-100 |               ER-AML                | Class-Incremental  |  ACC=0.24   | [ACC=0.24](https://openreview.net/forum?id=N8MaByOzUfb)                                                   | ✅             |
| Split CIFAR-100 |            Online Replay            | Class-Incremental  |  ACC=0.21   | ACC=0.21                                                                                                  | ✅ `Avalanche` |
|   Split MNIST   |                CoPE                 | Class-Incremental  |  ACC=0.93   | [ACC=0.93](https://arxiv.org/abs/2009.00919)                                                              | ✅             |
|   Split MNIST   |            Online Replay            | Class-Incremental  |  ACC=0.92   | ACC=0.92                                                                                                  | ✅ `Avalanche` |

## Python dependencies for experiments
Outside Python standard library, the main packages required to run the experiments are PyTorch, Avalanche and Pandas. 
* **Avalanche**: The latest version of this repo requires the latest Avalanche version (from master branch): `pip install git+https://github.com/ContinualAI/avalanche.git`. The CL baselines repo is tagged with the supported Avalanche version (you can browse the tags to check out all the versions). You can install the corresponding Avalanche versions with `pip install avalanche-lib==[version number]`, where `[version number]` is of the form `0.1.0`.
For some strategies (e.g., LaMAML) you may need to install Avalanche with extra packages, like `pip install avalanche-lib[extra]`. 
For more details on how to install Avalanche, please check out the complete guide [here](https://avalanche.continualai.org/getting-started/how-to-install). 
* **PyTorch**: we recommend to follow [the official guide](https://pytorch.org/get-started/locally/).
* **Pandas**: `pip install pandas`. [Official guide](https://pandas.pydata.org/docs/getting_started/install.html#installing-pandas).


## Run experiments with Python
Place yourself into the project root folder.

Experiments can be run with a python script by simply importing the function from the `experiments` folder and executing it.  
By default, experiments will run on GPU, when available.

The input argument to each experiment is an optional dictionary of parameters to be used in the experiments. If `None`, default
parameters (taken from original paper) will be used.

```python
from experiments.split_mnist import synaptic_intelligence_smnist  # select the experiment

 # can be None to use default parameters
custom_hyperparameters = {'si_lambda': 0.01, 'cuda': -1, 'seed': 3}

# run the experiment
result = synaptic_intelligence_smnist(custom_hyperparameters)

# dictionary of avalanche metrics
print(result)  
```

## Command line experiments
Place yourself into the project root folder.   
You should add the project root folder to your PYTHONPATH. 

For example, on Linux you can set it up globally:
```bash
export PYTHONPATH=${PYTHONPATH}:/path/to/continual-learning-baselines
```
or just for the current command:
```bash
PYTHONPATH=${PYTHONPATH}:/path/to/continual-learning-baselines command to be executed
```

You can run experiments directly through console with the default parameters.  
Open the console and run the python file you want by specifying its path.

For example, to run Synaptic Intelligence on Split MNIST: 
```bash
python experiments/split_mnist/synaptic_intelligence.py
```

To execute experiment with custom parameters, please refer to the previous section.


## Run tests
Place yourself into the project root folder.

You can run all tests with
```bash
python -m unittest
```

or you can specify a test by providing the test name in the format `tests.strategy_class_name.test_benchmarkname`.

For example to run Synaptic Intelligence on Split MNIST you can run:
```bash
python -m unittest tests.SynapticIntelligence.test_smnist
```

## Cite
If you used this repo you automatically used Avalanche, please remember to cite our reference paper published at the [CLVision @ CVPR2021](https://sites.google.com/view/clvision2021/overview?authuser=0) workshop: ["Avalanche: an End-to-End Library for Continual Learning"](https://arxiv.org/abs/2104.00405). 
This will help us make Avalanche better known in the machine learning community, ultimately making it a better tool for everyone:

```
@InProceedings{lomonaco2021avalanche,
    title={Avalanche: an End-to-End Library for Continual Learning},
    author={Vincenzo Lomonaco and Lorenzo Pellegrini and Andrea Cossu and Antonio Carta and Gabriele Graffieti and Tyler L. Hayes and Matthias De Lange and Marc Masana and Jary Pomponi and Gido van de Ven and Martin Mundt and Qi She and Keiland Cooper and Jeremy Forest and Eden Belouadah and Simone Calderara and German I. Parisi and Fabio Cuzzolin and Andreas Tolias and Simone Scardapane and Luca Antiga and Subutai Amhad and Adrian Popescu and Christopher Kanan and Joost van de Weijer and Tinne Tuytelaars and Davide Bacciu and Davide Maltoni},
    booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition},
    series={2nd Continual Learning in Computer Vision Workshop},
    year={2021}
}
```

## Contribute to the project
We are always looking for new contributors willing to help us in the challenging mission of providing robust experiments
to the community. Would you like to join us? The steps are easy!

1. Take a look at the opened issues and find yours
2. Fork this repo and write an experiment (see next section)
3. Submit a PR and receive support from the maintainers
4. Merge the PR, your contribution is now included in the project!


### Write an experiment
1. Create the appropriate script into `experiments/benchmark_folder`. If the benchmark is not present, you can add one.
2. Fill the `experiment.py` file with your code, following the style of the other experiments. The script should return the metrics used by the related test.
3. Add to `tests/target_results.csv` the expected result for your experiment. You can add a number or a list of numbers.
4. Write the unit test in `tests/strategy_folder/experiment.py`. Follow the very simple structure of existing tests.
5. Update table in `README.md`.


### Find the avalanche commit which produced a regression
1. Place yourself into the avalanche folder and make sure you are using the avalanche version from that repository 
in your python environment (it is usually enough to add `/path/to/avalanche` to your `PYTHONPATH`). 
2. Use the `gitbisect_test.sh` (provided in this repository) in combination with `git bisect` to retrieve the avalanche commit introducing the regression.  
`git bisect start HEAD v0.1.0 -- # HEAD (current version) is bad, v0.1.0 is good`  
`git bisect run /path/to/gitbisect_test.sh /path/to/continual-learning-baselines optional_test_name`  
`git bisect reset`
3. The `gitbisect_test.sh` script requires a mandatory parameter pointing to the `continual-learning-baselines`
directory and an optional parameter specifying the path to a particular unittest (e.g., `tests.EWC.test_pmnist`).
If the second parameter is not given, all the unit tests will be run.
4. The terminal output will tell you which commit introduced the bug
5. You can change the `HEAD` and `v0.1.0` ref to any avalanche commit.

