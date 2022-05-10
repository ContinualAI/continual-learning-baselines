import unittest
from tests.utils import get_target_result, get_average_metric
from experiments.split_cifar100 import icarl_scifar100


class iCARL(unittest.TestCase):
    """
        Reproducing iCaRL experiments from paper
        "iCaRL: Incremental Classifier and Representation Learning",
        Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert;
        Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2001-2010
        https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
    """

    def test_scifar100(self):
        """
            scifar100 with 10 batches
        """
        dict_iCaRL_aia = icarl_scifar100()
        avg_ia = get_average_metric(dict_iCaRL_aia)
        target_acc = get_target_result('iCaRL', 'scifar100')
        print(f"scifar100-batch=10 Average Incremental Accuracy: {avg_ia:.5f}")

        if target_acc > avg_ia:
            self.assertAlmostEqual(target_acc, avg_ia, delta=0.03)
