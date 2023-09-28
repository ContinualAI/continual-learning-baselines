import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_tiny_imagenet import packnet_stinyimagenet


class PackNet(unittest.TestCase):
    """
    Reproduce Delange et al. (2021) benchmark results for PackNet
    (Mallya & Lazebnik, 2018) on Split Tiny ImageNet

    Delange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A.,
      Slabaugh, G., & Tuytelaars, T. (2021). A continual learning survey: Defying
      forgetting in classification tasks. IEEE Transactions on Pattern Analysis
      and Machine Intelligence, 1–1. https://doi.org/10.1109/TPAMI.2021.3057446

    Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single
      Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer Vision
      and Pattern Recognition, 7765–7773. https://doi.org/10.1109/CVPR.2018.00810
    """

    def test_stinyimagenet(self):
        res = packnet_stinyimagenet({'seed': 0})
        avg_stream_acc = get_average_metric(res)
        print(f"PackNet-STinyImagenet Average Stream Accuracy: {avg_stream_acc:.2f}")

        target_acc = float(get_target_result('packnet', 'stiny-imagenet'))
        if target_acc > avg_stream_acc:
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
