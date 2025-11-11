# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import torch.nn as nn

from mmseg.models import SegformerHead, accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import SampleList

# Import our custom Hadamard encoder/decoder
from hadamard.hadamard_codec import HadamardCodec


@MODELS.register_module()
class SegformerHeadHadamard(SegformerHead):
    def __init__(self, hadamard_codec=dict(type="HadamardCodec", hadamard_size=(19,32)), hadamard_size=(19, 32), with_cp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_cp = with_cp
        self.hadamard_codec = MODELS.build(hadamard_codec)
        self.hadamard_size = hadamard_size

    def forward(self, inputs):
        if self.with_cp:
            def inner_forward(*args):
                return super(SegformerHeadHadamard, self).forward(args)

            return checkpoint(inner_forward, *inputs)
        else:
            return super(SegformerHeadHadamard, self).forward(inputs)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): Raw Hadamard logits from forward()
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples with both encoded and original labels.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_label_original          = self._stack_batch_gt(batch_data_samples)                                                  # [B, 1, H, W]
        seg_label_original          = seg_label_original.squeeze(1)                                                             # [B, H, W]

        # Encode original labels to Hadamard codes, get valid mask
        seg_label_hadamard_tanh, seg_label_hadamard_sigmoid, valid_mask  = self.hadamard_codec.encode(seg_label_original)       # [B, C, H, W]; [B, C, H, W]; [B, H, W]

        valid_mask = valid_mask.unsqueeze(1)                                                                                    # [B, 1, H, W]

        # Hadamard encoded pixel wise ignore weights
        seg_weight_hadamard = valid_mask.expand(-1, self.hadamard_size[1], -1, -1).float()                                      # [B, C, H, W]
        seg_weight_original = valid_mask.expand(-1, self.hadamard_size[0], -1, -1).float()                                      # [B, N, H, W]

        # Resize logits to match label size
        seg_logits = resize(
            input=seg_logits,
            size=seg_label_hadamard_sigmoid.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # Decode the resized logits to get all decoded values
        predictions, _, seg_logits_original, seg_logits_hadamard, error_vector_class = self.hadamard_codec.decode(seg_logits)   # [B, N, H, W]; [B, H, W]; [B, N, H, W]; [B, C, H, W]; [B, N, H, W]

        loss = dict()
 
        loss["error_vector_class"] = error_vector_class.abs().mean()
        loss['acc_seg'] = accuracy(predictions, seg_label_original, ignore_index=self.ignore_index)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        one_hot_space_optimization = ['hadamard_ce_loss','hadamard_gini_loss', 'hadamard_l1_loss', 'hadamard_mse_loss']
        hadamard_space_optimization = ['hadamard_mse_codes_loss', 'hadamard_l1_codes_loss', 'hadamard_bce_codes_loss']

        for loss_decode in losses_decode:
            if loss_decode.loss_name in one_hot_space_optimization:
                pred = predictions
                target = seg_label_original
                target_weight = seg_weight_original

                if 'gini' in loss_decode.loss_name:
                    target = None

                if 'ce' in loss_decode.loss_name:
                    if not self.hadamard_codec.use_simplex:
                        pred = seg_logits_original
                        
                    target_weight = None

                loss[loss_decode.loss_name] = loss_decode(
                    pred,
                    target,
                    weight=target_weight
                )

            elif loss_decode.loss_name in hadamard_space_optimization:
                pred = seg_logits_hadamard
                target = seg_label_hadamard_tanh
                target_weight = seg_weight_hadamard

                if 'bce' in loss_decode.loss_name:
                    pred = seg_logits
                    target = seg_label_hadamard_sigmoid

                loss[loss_decode.loss_name] = loss_decode(
                    pred,
                    target,
                    weight=target_weight
                )

        return loss