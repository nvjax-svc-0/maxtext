#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Normalization Layers."""

from typing import Any, Tuple, Optional
import functools

from flax import linen as nn
from jax import lax
import jax
import jax.numpy as jnp
from MaxText import max_logging
from MaxText.common_types import DecoderBlockType
from MaxText.layers.initializers import Initializer


class RMSNorm(nn.Module):
  """RMS normalization."""

  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[Optional[str], ...] = ()
  scale_init: Initializer = nn.initializers.ones
  parameter_memory_host_offload: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.param(
        "scale",
        nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
        (features,),
        self.weight_dtype,
    )
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, jax._src.sharding_impls.TransferToMemoryKind("device"))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale

from transformer_engine.jax.flax.module import LayerNorm
class TENormWrapper(LayerNorm):

  # This is used for casting the activations before they're passed to the
  # actual module
  jax_activation_dtype: jnp.dtype = jnp.float32

  # def init(*kwargs, **kwargs):

  # def __init__(self, *args, **kwargs):
  #   self.jax_activation_dtype = kwargs["dtype"]

  def __new__(cls, *args, **kwargs):
    ## Following attributes need to be converted to Transformer-Engine
    ## equivalent arguments.
    # epsilon
    # dtype
    # weight_dtype
    # kernel_axes
    # scale_init
    # use_bias
    # reductions_in_fp32

    ## TODO (sudhakars): Need to add assertion checks
    use_bias = kwargs.get("use_bias", False)
    weight_dtype = kwargs["weight_dtype"]
    dtype = kwargs["dtype"]
    epsilon = kwargs["epsilon"]
    name = kwargs["name"]

    ## wrong coz I realized layernorm without bias is not rmsnorm
    # norm_type = "layernorm" if use_bias else "rmsnorm"


    te_norm_kwargs = {
      "dtype": weight_dtype,
      "epsilon": epsilon,
      "name": name,
      "layernorm_type": norm_type,
    }

    # Initialize the module with appropriate flags
    te_norm_initialized = LayerNorm(**te_norm_kwargs)

    ## TODO (sudhakars): The following is apparently not allowed in JAX. (Call it dep1)
    # # Post initialization attributes setting
    # te_norm_initialized.jax_activation_dtype = dtype

    return te_norm_initialized

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    ## Can't do this due to dep1 above
    # x = jnp.asarray(x, self.jax_activation_dtype)
    return super().__call__(x)

def get_te_norm_wrapper(norm_type):
    from transformer_engine.jax.flax.module import LayerNorm

    def te_norm_wrapper(*args, **kwargs):
        ln_kwargs = {
          "dtype": kwargs['weight_dtype'],
          "epsilon": kwargs['epsilon'],
          "name": kwargs['name'],
        }
        norm_module_initialized = functools.partial(LayerNorm, layernorm_type=norm_type)(**ln_kwargs)

        def apply_te_norm(inputs):
          inputs = jnp.asarray(inputs, dtype=kwargs["dtype"])
          # return jnp.asarray(norm_module_initialized(inputs), dtype=kwargs["dtype"])
          ## In TE flax module, it's gauranteed that output type is the same as input type
          return norm_module_initialized(inputs)

        return apply_te_norm

    return te_norm_wrapper

def get_norm_layer(cfg):
  if cfg.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.GEMMA,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.LLAMA4,
    ):
    return get_te_norm_wrapper(norm_type="rmsnorm") if cfg.te_norm else RMSNorm
  elif cfg.decoder_block == DecoderBlockType.GPT3:
    from layers import gpt3

    # TODO (sudhakars): use_bias should be from config
    return get_te_norm_wrapper(norm_type="layernorm") if cfg.te_norm else \
      functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=True)
  else:
    raise ValueError(f"Incorrect decoder_block name {cfg.decoder_block=}")
