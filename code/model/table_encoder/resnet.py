from torch import nn
import torch.nn.functional as F 
from transformers.models.t5.modeling_t5 import T5LayerNorm
from einops import rearrange

class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_type == 'bart':
            layer_norm_eps = 1e-12
        elif config.model_type == 't5':
            layer_norm_eps = 1e-12
        else:
            layer_norm_eps = config.layer_norm_eps
        self.conv1 = nn.Conv2d(
            in_channels =config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size =(1, 1),
            padding=0
        )
        self.norm1 = T5LayerNorm(config.hidden_size, layer_norm_eps)

        self.conv2 = nn.Conv2d(
            in_channels =config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size =(3, 3),
            padding=1
        )
        self.norm2 = T5LayerNorm(config.hidden_size, layer_norm_eps)

        self.conv3 = nn.Conv2d(
            in_channels =config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size =(1, 1),
            padding=0
        )
        self.norm3 = T5LayerNorm(config.hidden_size, layer_norm_eps)

    def layer_forward(self, x, conv, norm):
        x = conv(x)
        n = x.size(-1)
        x = rearrange(x, 'b d m n -> b (m n) d')
        x = norm(x)
        x = F.relu(x)
        x = rearrange(x, 'b (m n) d -> b d m n', n=n)
        return x

    def forward(self, x_input, **kwargs):
        x = rearrange(x_input, 'b m n d -> b d m n')
        x = self.layer_forward(x, self.conv1, self.norm1)
        x = self.layer_forward(x, self.conv2, self.norm2)
        x = self.layer_forward(x, self.conv3, self.norm3)
        x = rearrange(x, 'b d m n -> b m n d')
        return x + x_input
