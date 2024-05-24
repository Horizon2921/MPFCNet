import functools
from typing import Any, Sequence, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp


Conv3x3x3 = functools.partial(nn.Conv, kernel_size=(3,3,3))
Conv1x1x1 = functools.partial(nn.Conv, kernel_size=(1,1,1))
ConvT_up = functools.partial(nn.ConvTranspose, kernel_size=(2,2,2), stride=(2,2,2))
Conv_down = functools.partial(nn.Conv, kernel_size=(4, 4, 4), stride=(2,2,2))
weight_initializer = nn.initializers.normal(stddev=2e-2)

class MlpBlock(nn.Module):
  """A 1-hidden-layer MLP block, applied over the last dimension."""
  mlp_dim: int
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, d = x.shape
    x = nn.Dense(self.mlp_dim, use_bias=self.use_bias,
                 kernel_init=weight_initializer)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn.Dense(d, use_bias=self.use_bias,
                 kernel_init=weight_initializer)(x)
    return x


class SELayer(nn.Module):
    features: int
    reduction: int = 4
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        # 3D global average pooling
        y = jnp.mean(x, axis=[1, 2, 2], keepdims=True)
        # Squeeze (in Squeeze-Excitation)
        y = Conv1x1x1(self.features // self.reduction, use_bias=self.use_bias)(y)
        y = nn.relu(y)
        # Excitation (in Squeeze-Excitation)
        y = Conv1x1x1(self.features, use_bias=self.use_bias)(y)
        y = nn.sigmoid(y)
        return x * y

def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x

class RCAB(nn.Module):
    "Residual c hannel attention block. Contains LN, Conv, LRelu, Conv, SELayer"
    fetures: int
    reduction: int = 4
    lrelu_slope: float = 0.2
    use_bias : bool = True

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = nn.LayerNorm(name="LayerNorm")(x)
        x = Conv3x3x3(features=self.features, use_bias=self.use_bias, name="conv1")(x)
        x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
        x = Conv3x3x3(features=self.features, use_bias=self.use_bias, name="conv2")(x)
        x = SELayer(features=self.features, reduction=self.reduction,
                    use_bias=self.use_bias, name="channel_attention")(x)
        return x + shortcut