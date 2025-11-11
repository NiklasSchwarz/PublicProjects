import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class HadamardMSELoss(nn.Module):
    """
    L2 loss for decoded network output y and original labels.
    This loss term reinforces the segmentation by including a L2 penalty
    """
    
    def __init__(self, 
                 reduction='mean',
                 loss_name='hadamard_mse_loss',
                 loss_weight=1.0,
                 ignore_index=255):

        super(HadamardMSELoss, self).__init__()
        self.loss_name = loss_name
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Computes the weighted MSE loss between predicted and ground truth Hadamard decoded outputs.
        with optional masking of ignored indices.
        Args:
            pred (Tensor): Predicted Hadamard codes of shape [N, num_classes, H, W] (decoded probs, resized).
            target (Tensor): Ground truth original labels of shape [N, H, W].
            weight (Tensor, optional): Optional tensor of weights for each sample.
            **kwargs: Additional keyword arguments. May include:
        Returns:
            Tensor: The computed loss value, reduced according to the specified reduction method and scaled by loss_weight.
        Notes:
            - If `self.ignore_index` is set, positions in `target` equal to this value are ignored in the loss computation.
            - The reduction method (`mean` or `sum`) is determined by `self.reduction`.
        """
        _, C, _, _ = pred.shape

        if hasattr(self, 'ignore_index'):
            target_clamped = target.clone()
            target_clamped[target_clamped == self.ignore_index] = 0  # Ersatz für one_hot
        else:
            target_clamped = target

        # One-hot Encoding des gecappten Targets
        target_one_hot = F.one_hot(target_clamped, num_classes=C).permute(0,3,1,2).float()

        criterion = nn.MSELoss(reduction='none')
        loss = criterion(pred, target_one_hot)

        if weight is not None:
            loss = loss * weight

            if self.reduction == 'mean':
                denom = (weight.sum()).clamp(min=1.0)
                loss = loss.sum() * weight.shape[1] / denom
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return self.loss_weight * loss