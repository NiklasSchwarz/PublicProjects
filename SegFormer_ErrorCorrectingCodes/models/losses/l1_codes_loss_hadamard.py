import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class HadamardCodesL1Loss(nn.Module):
    """
    MSE loss for Hadamard encoded predictions.
    This loss term provides direct supervision in Hadamard space by comparing
    predicted Hadamard codes with ground truth Hadamard-encoded labels.
    Matches the paper's mse_codes_loss component.
    """
    
    def __init__(self, 
                 reduction='mean',
                 loss_name='hadamard_mse_codes_loss',
                 loss_weight=1.0,
                 ignore_index=255):

        super(HadamardCodesL1Loss, self).__init__()
        self.loss_name = loss_name
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred: Predicted Hadamard codes [N, hadamard_size, H, W]
            target: Hadamard encoded ground truth class indices  [N, hadamard_size, H, W]
            **kwargs: Additional arguments including:
                - weight: Optional tensor of weights
        """
        criterion = nn.L1Loss(reduction='none')

        # Apply MSE loss directly on Hadamard codes
        loss = criterion(pred, target)

        if weight is not None:
            loss = loss * weight

            if self.reduction == 'mean':
                denom = weight.sum().clamp(min=1.0)
                loss = loss.sum() / denom
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return self.loss_weight * loss