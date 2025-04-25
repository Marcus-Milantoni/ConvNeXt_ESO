import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops.misc import Permute, Conv3dNormActivation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param

from typing import Callable, Optional, Any, List, Sequence
from functools import partial


class LayerNorm3D(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # Reshape the input tensor to (batch_size, num_channels, depth, height, width)
        x = x.permute(0,2,3,4,1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0,4,1,2,3)
        return x
    

class CNBlock3D(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale: float,
            stochastic_depth_prob: float,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
        ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 4, 1]),
            norm_layer(dim),
            nn.Linear(in_features = dim, out_features=4*dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features = 4*dim, out_features=dim, bias=True),
            Permute([0, 4, 1, 2, 3])
        )
        self.layer_scale = nn.Parameter(layer_scale * torch.ones((dim, 1, 1, 1)))
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, input: Tensor) -> Tensor:
        x = self.layer_scale * self.block(input)
        x = self.stochastic_depth(x)
        x += input
        return x
    

class CNBlockConfig:
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            num_layers: int
    ) -> None:
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        return f"CNBlockConfig(input_channels={self.input_channels}, output_channels={self.output_channels}, num_layers={self.num_layers})"


class ConvNeXt_ESO_adapt(nn.Module):
    def __init__(
            self,
            block_settings: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.0,
            in_channels: int = 2,
            layer_scale: float = 1e-6,
            num_classes: int = 2,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
    ) -> None:
        
        super().__init__()
        
        if not block_settings:
            raise ValueError("block_settings cannot be empty.")
        elif not (isinstance(block_settings, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_settings])):
                  raise TypeError("block_settings must be a list of CNBlockConfig instances.")
        
        if block is None:
            block = CNBlock3D
        if norm_layer is None:
            norm_layer = partial(LayerNorm3D, eps=1e-6)

        layers: List[nn.Module] = [] # The variable layers is a list of nn.Modules initialized to an empty list.
        
        # Stewm, which processes the input data before passing i to the main body
        first_conv_out_channels = block_settings[0].input_channels
        layers.append(
            Conv3dNormActivation(
                in_channels, # Input channels for the model
                first_conv_out_channels,
                kernel_size = 4,
                stride = 4,
                padding = 0,
                norm_layer = norm_layer,
                activation_layer = None, 
                bias = True,
            )
        )

        total_stage_blocks  = sum([config.num_layers for config in block_settings])
        stage_block_idx = 0
        for config in block_settings:
            stage: List[nn.Module] = []

            for layer_idx in range(config.num_layers):
                sd_prob = stochastic_depth_prob * (stage_block_idx + layer_idx) / (total_stage_blocks - 1.0)
                stage.append(
                    block(
                        config.input_channels,
                        layer_scale=layer_scale,
                        stochastic_depth_prob=sd_prob
                    )
                )
                stage_block_idx += 1
            layers.append(nn.Sequential(*stage))
            
            if config.output_channels is not None:
                layers.append(
                    nn.Sequential(
                        norm_layer(config.input_channels),
                        nn.Conv3d(config.input_channels, config.output_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        last_block = block_settings[-1]
        last_conv_out_channels = (
            last_block.output_channels if last_block.output_channels is not None else last_block.input_channels
        )
        self.model_head = nn.Sequential(
            norm_layer(last_conv_out_channels),
            nn.Flatten(1),
            nn.Linear(last_conv_out_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_imp(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.model_head(x)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        return self.forward_imp(x)
        

def _convnext(
        block_settings: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        weights: Optional[WeightsEnum] = None,
        progress: bool = True,
        **kwargs: Any
) -> ConvNeXt_ESO_adapt:

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt_ESO_adapt(
        block_settings=block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs
    )

    return model


@register_model()
def convNeXt_tiny_ESO(*, progress: bool = True, **kwargs: Any) -> ConvNeXt_ESO_adapt:
    """ConvNeXt Tiny model for 3D medical image classification.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        ConvNeXt_ESO_adapt: The ConvNeXt Tiny model.
    """
    
    block_settings = [
        CNBlockConfig(input_channels=96, output_channels=192, num_layers=3),
        CNBlockConfig(input_channels=192, output_channels=384, num_layers=3),
        CNBlockConfig(input_channels=384, output_channels=786, num_layers=9),
        CNBlockConfig(input_channels=786, output_channels=None, num_layers=3),
    ]

    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)

    return _convnext(
        block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=None,
        progress=progress,
        **kwargs
    )


@register_model()
def custom_convnet_3s(*, progress:bool=True, **kwargs:Any) -> ConvNeXt_ESO_adapt:
    """Custom ConvNeXt model for 3D medical image classification.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        ConvNeXt_ESO_adapt: The custom ConvNeXt model.
    """
    
    block_settings = [
        CNBlockConfig(input_channels=96, output_channels=192, num_layers=3),
        CNBlockConfig(input_channels=192, output_channels=384, num_layers=3),
        CNBlockConfig(input_channels=384, output_channels=786, num_layers=3),
        CNBlockConfig(input_channels=786, output_channels=None, num_layers=3),
    ]

    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)

    return _convnext(
        block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=None,
        progress=progress,
        **kwargs
    )


@register_model()
def custom_convnet_33(*, progress:bool=True, **kwargs:Any) -> ConvNeXt_ESO_adapt:
    """Custom ConvNeXt model for 3D medical image classification.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        ConvNeXt_ESO_adapt: The custom ConvNeXt model.
    """
    
    block_settings = [
        CNBlockConfig(input_channels=96, output_channels=192, num_layers=3),
        CNBlockConfig(input_channels=192, output_channels=384, num_layers=3),
    ]

    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)

    return _convnext(
        block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=None,
        progress=progress,
        **kwargs
    )


@register_model()
def custom_convnet_336(*, progress:bool=True, **kwargs:Any) -> ConvNeXt_ESO_adapt:
    """Custom ConvNeXt model for 3D medical image classification.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        ConvNeXt_ESO_adapt: The custom ConvNeXt model.
    """
    
    block_settings = [
        CNBlockConfig(input_channels=96, output_channels=192, num_layers=3),
        CNBlockConfig(input_channels=192, output_channels=384, num_layers=3),
        CNBlockConfig(input_channels=384, output_channels=None, num_layers=6),
    ]

    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)

    return _convnext(
        block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=None,
        progress=progress,
        **kwargs
    )


@register_model()
def custom_convnet_333(*, progress:bool=True, **kwargs:Any) -> ConvNeXt_ESO_adapt:
    """Custom ConvNeXt model for 3D medical image classification.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        ConvNeXt_ESO_adapt: The custom ConvNeXt model.
    """
    
    block_settings = [
        CNBlockConfig(input_channels=96, output_channels=192, num_layers=3),
        CNBlockConfig(input_channels=192, output_channels=384, num_layers=3),
        CNBlockConfig(input_channels=384, output_channels=None, num_layers=3),
    ]

    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)

    return _convnext(
        block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=None,
        progress=progress,
        **kwargs
    )


@register_model()
def convNeXt_2s(*, progress: bool = True, **kwargs: Any) -> ConvNeXt_ESO_adapt:
    """ConvNeXt Small model for 3D medical image classification.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        ConvNeXt_ESO_adapt: The ConvNeXt Small model.
    """
    
    block_settings = [
        CNBlockConfig(input_channels=96, output_channels=192, num_layers=2),
        CNBlockConfig(input_channels=192, output_channels=384, num_layers=2),
        CNBlockConfig(input_channels=384, output_channels=786, num_layers=2),
        CNBlockConfig(input_channels=786, output_channels=None, num_layers=2),
    ]

    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)

    return _convnext(
        block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=None,
        progress=progress,
        **kwargs
    )

@register_model()
def convNeXt_small_ESO(*, progress: bool = True, **kwargs: Any) -> ConvNeXt_ESO_adapt:
    """ConvNeXt Small model for 3D medical image classification.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        ConvNeXt_ESO_adapt: The ConvNeXt Small model.
    """
    
    block_settings = [
        CNBlockConfig(input_channels=96, output_channels=192, num_layers=3),
        CNBlockConfig(input_channels=192, output_channels=384, num_layers=3),
        CNBlockConfig(input_channels=384, output_channels=786, num_layers=27),
        CNBlockConfig(input_channels=786, output_channels=None, num_layers=3),
    ]

    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)

    return _convnext(
        block_settings,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=None,
        progress=progress,
        **kwargs
    )

