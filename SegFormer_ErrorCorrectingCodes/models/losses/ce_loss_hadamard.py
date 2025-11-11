import torch.nn as nn
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class HadamardCrossEntropyLoss(nn.Module):
    """
    Multi-class cross-entropy loss for Hadamard encoded predictions.
    
    This loss encourages the predicted segmentation probabilistic map ŷ 
    (decoded from Hadamard codes) to be similar to the one-hot encoded ground truth y.
    
    Process:
    1. Decode Hadamard predictions to class probabilities using H(z) = softmax(H^T * tanh(z))
    2. Apply cross-entropy loss between decoded probabilities and ground truth class indices
    """
    
    def __init__(self, 
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='hadamard_ce_loss',
                 ignore_index=255,
                 use_simplex=False):
                 
        super(HadamardCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.ignore_index = ignore_index
        self.use_simplex = use_simplex

    def forward(self, pred, target, weight, **kwargs):
        """
        Args:
            y: torch.Size([1, NUM_CLASSES, H, W])
            target: Ground truth class indices [1, H, W]
            **kwargs: Additional arguments including:
                - weight: Optional tensor of weights
        """        
        if self.use_simplex:
            eps = 1e-6
            pred = pred.clamp(min=eps, max=1.0 - eps).contiguous()
            loss = F.nll_loss(torch.log(pred), target, reduction=self.reduction, ignore_index=self.ignore_index)
        else:
            # Apply cross entropy loss using decoded probabilities
            loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction=self.reduction)

        return self.loss_weight * loss