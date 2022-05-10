from pathlib import Path
import inspect
from pandas import read_csv
import os
import tests


def pandas_to_list(input_str):
    return [float(el) for el in input_str.strip('[] ').split(' ')]


def get_target_result(strat_name: str, bench_name: str):
    """
    Read the target_results.csv file and retrieve the target performance for
    the given strategy on the given benchmark.
    :param strat_name: strategy name as found in the target file
    :param bench_name: benchmark name as found in the target file
    :return: target performance (either a float or a list of floats)
    """

    p = os.path.join(Path(inspect.getabsfile(tests)).parent, 'target_results.csv')
    data = read_csv(p, sep=',', comment='#')
    target = data[(data['strategy'] == strat_name) & (data['benchmark'] == bench_name)]['result'].values[0]
    if isinstance(target, str) and target.startswith('[') and target.endswith(']'):
        target = pandas_to_list(target)
    else:
        target = float(target)
    return target


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
