import unittest
from tests.utils import get_target_result, get_average_metric
from experiments.permuted_mnist import lfl_pmnist


class LFL(unittest.TestCase):
    """
        Reproducing Less Forgetful Learning experiments
        "Less-forgetting Learning in Deep Neural Networks"
        Heechul Jung, Jeongwoo Ju, Minju Jung and Junmo Kim;
        arXiv, 2016, https://arxiv.org/pdf/1607.00122.pdf
    """

    def test_pmnist(self):
        res = lfl_pmnist({'seed': 0})
        exps_acc = []
        for k, v in res.items():
            if k.startswith('Top1_Acc_Exp'):
                exps_acc.append(v)
        target_acc = get_target_result('lfl', 'pmnist')
        print(f"LFL-PMNIST Experiences Accuracy: {exps_acc}")

        # each experience accuracy should be at least target acc
        for el in exps_acc:
            if target_acc > el:
                self.assertAlmostEqual(target_acc, el, delta=0.03)
