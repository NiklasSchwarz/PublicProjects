import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class HadamardGiniLoss(nn.Module):
    """
    Gini coefficient loss for Hadamard encoded predictions.

    This loss term reinforces the segmentation by including a Gini coefficient penalty for the differences between ˆy and y,
    where ˆy are the predicted probabilities from softmax(H^T * tanh(z)) and y are the ground truth one-hot vectors.
    
    Process:
    1. Decode Hadamard predictions to class probabilities using H(z) = softmax(H^T * tanh(z))
    2. Apply Gini coefficient loss between probabilities and one-hot ground truth directly
    """
    
    def __init__(self, 
                 reduction='mean',
                 loss_name='hadamard_gini_loss',
                 loss_weight=1.0,
                 ignore_index=255):

        super(HadamardGiniLoss, self).__init__()
        self.loss_name = loss_name
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred: torch.Size([1, NUM_CLASSES, H, W])
            target: Ground truth class indices [N, H, W] or [N, 1, H, W]
            weight: Optional tensor of weights for each sample
            **kwargs: Additional arguments including:
                - weight: Optional tensor of weights
        """
        
        # Compute Gini coefficient loss (entropy penalty) - matches paper implementation
        # Paper uses: gn_loss = -tf.reduce_mean(tf.square(gen_prob))
        # Gini coefficient: 1 - sum(p^2) over classes
        loss = pred ** 2

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

        return self.loss_weight * loss * (-1)    