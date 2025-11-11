import torch
import torch.nn.functional as F
import math
import numpy as np
from mmengine import MODELS
import torch.nn as nn
from scipy.linalg import hadamard
from torch.utils.checkpoint import checkpoint


@MODELS.register_module()
class HadamardCodec(nn.Module):
    """
    A unified class for Hadamard encoding and decoding operations.
    
    This class handles:
    1. Encoding class labels to Hadamard codes
    2. Decoding Hadamard predictions back to class probabilities/predictions
    
    Args:
        num_classes (int): Number of actual classes (e.g., 19)
        hadamard_size (tuple): Size of the Hadamard matrix (num_classes, code_length)
        ignore_index (int): Label to ignore during encoding/decoding
        use_simplex (bool): Whether to use simplex projection during decoding
        activation_function (str): Activation function to use ('tanh' or 'sigmoid')
        use_all_one_codeword (bool): Whether to include the all-one codeword
        with_cp (bool): Whether to use checkpointing for memory efficiency
        augmented (bool): Whether to augment the Hadamard matrix by adding negative rows
    """

    def __init__(self, 
                 hadamard_size=(19, 32), 
                 ignore_index=255, 
                 use_simplex=False, 
                 activation_function="tanh",
                 use_all_one_codeword=True, 
                 with_cp=False, 
                 augmented=False):
        
        super().__init__()  

        self.num_classes = hadamard_size[0]
        self.output_channels = hadamard_size[1]
        self.use_simplex = use_simplex
        self.activation_function = activation_function
        self.with_cp = with_cp
        self.ignore_index = ignore_index

        full_codes_matrix = hadamard(self.output_channels).astype(np.float32)

        if not use_all_one_codeword:
            if augmented: 
                full_codes_matrix = np.vstack((full_codes_matrix, -full_codes_matrix[1:]))

            self.codes_matrix = full_codes_matrix[1:self.num_classes+1, :]                   # 19x32
        else:
            if augmented: 
                full_codes_matrix = np.vstack((full_codes_matrix, -full_codes_matrix[:]))

            self.codes_matrix = full_codes_matrix[:self.num_classes, :]                      # 19x32      

        codes_tensor = torch.tensor(self.codes_matrix, dtype=torch.float32)
        codes_transpose = codes_tensor.t().contiguous()                                      # 32x19      

        self.register_buffer('codes_tensor', codes_tensor, persistent=False)
        self.register_buffer('codes_transpose_tensor', codes_transpose, persistent=False)

    def encode(self, target):
        """
        Encode class labels to Hadamard codes.
        
        Args:
            target: Class labels [H, W] with values 0-18 for 19 classes
            
        Returns:
            label_hadamard_tanh: Hadamard codes scaled to [-1, 1]
            label_hadamard_sigmoid: Hadamard codes scaled to [0, 1]
            valid_mask: Mask indicating valid pixels (not ignore_index)
        """
        if isinstance(target, np.ndarray) and not target.flags.c_contiguous:
            target = np.ascontiguousarray(target)

        device = self.codes_tensor.device 
        target_tensor = torch.as_tensor(target, dtype=torch.long, device=device)    
        codes = torch.zeros((*target_tensor.shape, self.output_channels), dtype=torch.float32, device=device)

        valid_mask = (target_tensor != self.ignore_index)

        # Encode only valid pixels with Hadamard codes [-1, 1]

        codes[valid_mask] = self.codes_tensor[target_tensor[valid_mask]]
        label_hadamard_tanh = codes.clone()

        # Scale Hadamard codes to [0, 1] for sigmoid activation
        codes[valid_mask] = (codes[valid_mask] + 1.0) / 2.0
        label_hadamard_sigmoid = codes

        # Reshape to [N, C, H, W] for loss computation
        label_hadamard_tanh = label_hadamard_tanh.permute(0, 3, 1, 2).contiguous()
        label_hadamard_sigmoid = label_hadamard_sigmoid.permute(0, 3, 1, 2).contiguous()

        return label_hadamard_tanh, label_hadamard_sigmoid, valid_mask

    def decode(self, seg_logits):
        """
        Decode Hadamard predictions to class predictions.
        
        Args:
            seg_logits: [N, hadamard_size, H, W] Hadamard predictions from network
            
        Returns:
            probabilities: [N, num_classes, H, W] Class probabilities
            mask: [N, H, W] Predicted class indices
            seg_logits_original: [N, num_classes, H, W] Projected raw logits
            seg_logits_hadamard: [N, hadamard_size, H, W] Activated Hadamard logits
            error_vector_class: [N, num_classes, H, W] Error vector between projected and raw logits
        """
        N, _, H, W = seg_logits.shape

        # Apply tanh to get predictions in (-1, 1) range like Hadamard values
        if self.activation_function == "sigmoid":
            seg_logits_hadamard = 2 * torch.sigmoid(seg_logits) - 1  
        elif self.activation_function == "tanh":
            seg_logits_hadamard = torch.tanh(seg_logits)
        else:
            raise NotImplementedError()

        # seg_logits: [N, hadamard_size, H, W] -> [N, H, W, hadamard_size]
        seg_logits = seg_logits_hadamard.permute(0, 2, 3, 1)

        # Matrix multiplication
        # [N, H, W, hadamard_size] x [hadamard_size, num_classes] = [N, H, W, num_classes]
        if self.use_simplex:
            seg_logits_original = torch.matmul(seg_logits, self.codes_transpose_tensor) / self.output_channels
        else:
            seg_logits_original = torch.matmul(seg_logits, self.codes_transpose_tensor)
            
        # Compute probabilities
        if not self.use_simplex:
            # Using projection onto probability simplex

            # Reshape back to [N, num_classes, H, W] for the raw logits output
            seg_logits_original = seg_logits_original.permute(0, 3, 1, 2)  # [N, num_classes, H, W]
            probabilities       = F.softmax(seg_logits_original, dim=1)  # [N, num_classes, H, W]
            error_vector_class  = torch.zeros_like(probabilities, device=probabilities.device)
        else:
            # Using Softmax
            seg_logits_original = seg_logits_original.contiguous().view(N * H * W, self.num_classes)
            if self.with_cp:
                probabilities = checkpoint(self.project_onto_simplex_torch, seg_logits_original)
            else:
                probabilities = self.project_onto_simplex_torch(seg_logits_original)

            error_vector_class = (probabilities - seg_logits_original).view(N, H, W, self.num_classes).permute(0, 3, 1,2).contiguous()
            probabilities = probabilities.view(N, H, W, self.num_classes).permute(0, 3, 1, 2).contiguous()  # [N, num_classes, H, W]
            seg_logits_original = seg_logits_original.view(N, H, W, self.num_classes).permute(0, 3, 1, 2).contiguous()  # [N, num_classes, H, W]


        # Use argmax as the prediction results
        mask = probabilities.argmax(dim=1)  # [N, H, W]

        return probabilities, mask, seg_logits_original, seg_logits_hadamard, error_vector_class

    def project_onto_simplex_torch(self, v):
        """
        Projects the input tensor onto the probability simplex.

        Args:
            v (tensor): Input tensor of shape (batch_size, n)

        Returns:
            tensor: Projected tensor of the same shape as input
        """
        # v: (batch_size, n)
        sorted_v, _ = torch.sort(v, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_v, dim=-1) - 1
        k = torch.arange(1, v.shape[1] + 1, device=v.device).view(1, -1)
        cond = sorted_v - cumsum / k > 0
        rho = cond.sum(dim=1) - 1
        theta = cumsum[torch.arange(v.shape[0]), rho] / (rho + 1)
        theta = theta.contiguous()
        q_proj = torch.clamp(v - theta.unsqueeze(1), min=0).contiguous()
        return q_proj

    def get_codes_matrix(self):
        """Return the codes matrix for inspection."""
        return self.codes_matrix

    def get_codes_transpose(self):
        """Return the transpose codes matrix for inspection."""
        return self.codes_transpose

    def to_device(self, device):
        """Move tensors to specified device."""
        self.codes_tensor = self.codes_tensor.to(device)
        self.codes_transpose_tensor = self.codes_transpose_tensor.to(device)
        return self

    def __repr__(self):
        return (f"HadamardCodec(hadamard_size={self.hadamard_size}, "
                f"num_classes={self.num_classes}, "
                f"matrix_shape={self.codes_matrix.shape})")
