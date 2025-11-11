import torch
from torch import Tensor
from mmengine.structures import PixelData

from mmseg.models import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (OptSampleList, SampleList)

# Import our custom Hadamard encoder/decoder
from hadamard.hadamard_codec import HadamardCodec

@MODELS.register_module()
class EncoderDecoderHadamard(EncoderDecoder):
    def __init__(self, hadamard_codec=dict(type="HadamardCodec", hadamard_size=(19,32)), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hadamard_codec = MODELS.build(hadamard_codec)

    def postprocess_result(self,
                           seg_logits: Tensor, 
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """

        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom = \
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                               padding_top:H - padding_bottom,
                               padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3,))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2,))

                # resize as original shape - keep batch dimension for decode
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            else:
                # Add batch dimension to match decode method requirements
                i_seg_logits = seg_logits[i:i + 1]

            # Decode Hadamard logits to class predictions
            i_seg_logits = i_seg_logits.to(torch.float32)
            alpha, mask, _, _, error_vector_class = self.hadamard_codec.decode(i_seg_logits)

            gt_sem_seg = data_samples[i].gt_sem_seg.data

            data_samples[i].set_data({
                'seg_logits':
                    PixelData(**{'data': alpha.squeeze(0)}),   # Hadamard coefficients [C, H, W]
                'pred_sem_seg':
                    PixelData(**{'data': mask}),               # predicted class mask [H, W]
                'gt_sem_seg': 
                    PixelData(**{'data': gt_sem_seg})
            })

        return data_samples