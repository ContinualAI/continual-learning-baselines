import torch.nn as nn
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from types import SimpleNamespace
import strategies
import os
from pathlib import Path
import inspect
from pandas import read_csv


def get_target_result(strat_name: str, bench_name: str):
    """
    Read the target_results.csv file and retrieve the target performance for
    the given strategy on the given benchmark.
    :param strat_name: strategy name as found in the target file
    :param bench_name: benchmark name as found in the target file
    :return: target performance (either a float or a list of floats)
    """

    p = os.path.join(Path(inspect.getabsfile(strategies)).parent, 'target_results.csv')
    data = read_csv(p, sep=',', comment='#')
    target = data[(data['strategy'] == strat_name) & (data['benchmark'] == bench_name)]['result'].values[0].strip()
    if isinstance(target, str) and target.startswith('[') and target.endswith(']'):
        target = pandas_to_list(target)
    else:
        target = float(target)
    return target


def pandas_to_list(input_str):
    return [float(el) for el in input_str.strip('[]').split(' ')]


def get_average_metric(metric_dict: dict, metric_name: str = 'Top1_Acc_Stream'):
    """
    Compute the average of a metric based on the provided metric name.
    The average is computed across the instance of the metrics containing the
    given metric name in the input dictionary.
    :param metric_dict: dictionary containing metric name as keys and metric value as value.
        This dictionary is usually returned by the `eval` method of Avalanche strategies.
    :param metric_name: the metric name (or a part of it), to be used as pattern to filter the dictionary
    :return: a number representing the average of all the metric containing `metric_name` in their name
    """

    avg_stream_acc = []
    for k, v in metric_dict.items():
        if k.startswith(metric_name):
            avg_stream_acc.append(v)
    return sum(avg_stream_acc) / float(len(avg_stream_acc))


def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    args.__dict__['check'] = True
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args


class MultiHeadMLP(MultiTaskModule):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 drop_rate=0):
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size)

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0):
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x


class SI_CNN(MultiTaskModule):
    def __init__(self, hidden_size=512):
        super().__init__()
        layers = nn.Sequential(*(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Flatten(),
                                 nn.Linear(2304, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5)
                                 ))
        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size, initial_out_features=10)

    def forward(self, x, task_labels):
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


__all__ = ['MultiHeadMLP', 'MLP', 'SI_CNN', 'get_average_metric',
           'create_default_args', 'get_target_result']
