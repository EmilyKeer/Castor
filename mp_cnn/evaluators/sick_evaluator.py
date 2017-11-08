from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

from mp_cnn.evaluators.general_multiclass_evaluator import GeneralMulticlassEvaluator


class SICKEvaluator(GeneralMulticlassEvaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(SICKEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)
