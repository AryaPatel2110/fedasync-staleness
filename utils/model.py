# Utilities for building models and converting parameters
from typing import Dict, List
import torch
import torch.nn as nn
from torchvision import models


def build_squeezenet(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Create SqueezeNet v1.1 and replace the classifier head."""
    if pretrained:
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    else:
        m = models.squeezenet1_1(weights=None)
    m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
    m.num_classes = num_classes
    return m


def state_to_list(state: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Flatten a state_dict to a list of tensors on CPU."""
    return [t.detach().cpu().clone() for _, t in state.items()]


def list_to_state(template: Dict[str, torch.Tensor], arrs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rebuild a state_dict from a list of tensors using a template for keys/dtypes/devices."""
    out: Dict[str, torch.Tensor] = {}
    for (k, v), a in zip(template.items(), arrs):
        out[k] = a.to(v.device).type_as(v)
    return out
