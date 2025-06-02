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

"""Linear Layers."""

import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp

from jax import lax
from jax.ad_checkpoint import checkpoint_name

import flax.linen as nn

from MaxText import max_logging
from MaxText.common_types import DecoderBlockType, DType, Array, Config
from MaxText.layers import quantizations
from MaxText.layers.normalizations import RMSNorm, get_norm_layer
from MaxText.layers.initializers import NdInitializer, nd_dense_init, default_bias_init
from MaxText.layers.quantizations import AqtQuantization as Quant


def _convert_to_activation_function(fn_or_string: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


def _compute_dot_general(inputs, kernel, kernel_axes, axis, contract_ind, matmul_precision, quant):
  """Computes a dot_general operation that may be quantized."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant:
    dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
    dot_general = dot_general_cls()
    return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
  return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation.
    quant: quantization config, defaults to None implying no quantization.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  kernel_axes: Tuple[Optional[str], ...] = ()
  quant: Optional[Quant] = None
  use_bias: bool = False
  matmul_precision: str = "default"
  parameter_memory_host_offload: bool = False
  dot_kernel: str = "te_dot"

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))

    if self.dot_kernel == "te_dot":
      # This import is only meant to work in a GPU build
      from transformer_engine.jax.flax.module import DenseGeneral as TEDenseGeneral

      def TE_kernel_init(rng, shape, dtype):
        return self.kernel_init(rng, shape, dtype, kernel_in_axis, kernel_out_axis)

      # NOTE: This is not needed since TE should be aware of the provided axis and not the
      #       other way around. But keeping this here to confirm before removing.
      # def replace_kernel_axes(kernel_axes):
      #   from transformer_engine.jax.sharding import W_FSDP_AXES, W_TP_AXES, HIDDEN_TP_AXES, HEAD_AXES
      #   maxtext_to_te_axes = {
      #     "embed": W_FSDP_AXES,
      #     "mlp": W_TP_AXES,
      #     "kv_head_dim": HIDDEN_TP_AXES,
      #     "kv": HIDDEN_TP_AXES,
      #     "kv_heads": HEAD_AXES,
      #     "q_heads": HEAD_AXES,
      #     "heads": HEAD_AXES,
      #   }
      #   return _canonicalize_tuple(maxtext_to_te_axes[ax] for ax in kernel_axes)

      # TODO: self.quant is not handled yet
      te_dense_general = TEDenseGeneral(
          features=self.features,
          kernel_init=TE_kernel_init,
          kernel_axes=self.kernel_axes,
          use_bias=self.use_bias,
          axis=self.axis,
          dtype=self.weight_dtype,
          name=self.name,
          # dtype=self.dtype,
      )
      # Cast the inputs into required activation dtype.
      inputs = jnp.asarray(inputs, dtype=self.dtype)
      output = te_dense_general(inputs)
    else:
      if quantizations.in_serve_mode(self.quant):
        # During aqt convert state we delete kernel weight from params to save memory.
        # Instead they are retrieved from the tensors stored in the 'aqt' collection.
        kernel = jnp.zeros(kernel_shape)
      kernel = self.param(
          "kernel",
          nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
          kernel_shape,
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
      # Move logit_dense kernel to device if parameter offloading is enabled
      if self.parameter_memory_host_offload:
        max_logging.log("linear.py: Moving parameter logits_dense kernel to device")
        kernel = jax.device_put(kernel, jax._src.sharding_impls.TransferToMemoryKind("device"))
      kernel = jnp.asarray(kernel, self.dtype)
      contract_ind = tuple(range(0, len(axis)))
      output = _compute_dot_general(inputs, kernel, self.kernel_axes, axis, contract_ind, self.matmul_precision, self.quant)

    if self.use_bias:
      bias_axes, bias_shape = (
          self.kernel_axes[-len(features) :],
          kernel_shape[-len(features) :],
      )
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(default_bias_init, bias_axes),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output

from transformer_engine.jax.flax.module import LayerNormMLP
from transformer_engine.jax.flax.module import DenseGeneral as TEDenseGeneral
import inspect

class TENormMLPWrapper(LayerNormMLP):
  def __new__(cls, *args, **kwargs):

    signature = inspect.signature(LayerNormMLP.__init__)
    lnmlp_expected_args = {
      key:value for key,value in kwargs.items() if key in signature
    }

    te_lnmlp_initialized = LayerNormMLP(**lnmlp_expected_args)

    return te_lnmlp_initialized

  @nn.compact
  def __call__(self, inputs: Array, deterministic: bool = False):
    ## Can't do this due to dep1 above
    # x = jnp.asarray(x, self.jax_activation_dtype)
    return super().__call__(inputs, deterministic=deterministic)

## While the above class based wrapper doesn't work, use this
def get_mlp_block(cfg):
  def te_mlp_block_wrapper(*args, **kwargs):
    signature = inspect.signature(LayerNormMLP.__init__)
    lnmlp_expected_args = {
      key:value for key,value in kwargs.items() if key in signature.parameters
    }
    lnmlp_expected_args["dtype"] = kwargs["weight_dtype"]
    lnmlp_expected_args["layernorm_type"] = "rmsnorm"
    lnmlp_expected_args["kernel_axes_1"] = ("embed", None, "mlp")
    lnmlp_expected_args["kernel_axes_2"] = ("mlp", "embed")
    lnmlp_expected_args["return_layernorm_output"] = False
    te_lnmlp_initialized = LayerNormMLP(**lnmlp_expected_args)


    def apply_te_mlp(inputs: Array, deterministic: bool = False):
      inputs = jnp.asarray(inputs, dtype=kwargs["dtype"])
      outputs = te_lnmlp_initialized(inputs, deterministic=deterministic)
      return outputs[0]

    return apply_te_mlp

  return te_mlp_block_wrapper if cfg.te_mlp else MlpBlock

## For TE DenseGeneral, kernel_init doesn't work since it needs additional
## arguments - kernel_in_axis and kernel_out_axis that are not accessible
## in this wrapper function. Need to think more on this.
def get_dense_general(cfg):
  def te_dense_general_wrapper(*args, **kwargs):
    signature = inspect.signature(TEDenseGeneral.__init__)
    dense_expected_args = {
      key:value for key,value in kwargs.items() if key in signature.parameters
    }
    # dense_expected_args["features"] = kwargs["features"]
    dense_expected_args["dtype"] = kwargs["weight_dtype"]

    te_dense_general_initialized = TEDenseGeneral(*args, **dense_expected_args)

    # te_dense_general_initialized

    def apply_te_dense_general(inputs: Array, deterministic: bool = False):
      inputs = jnp.asarray(inputs, dtype=kwargs["dtype"])
      outputs = te_dense_general_initialized(inputs, deterministic=deterministic)
      return outputs[0]

    return apply_te_dense_general

  return te_dense_general_wrapper if cfg.te_dense_general else DenseGeneral

class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: computation data type for the dense layer.
    weight_dtype: weight data type for the dense layer.
    use_bias: whether to add bias in all feedforward layers.
    use_pre_norm: whether to add pre layer norm in mlp layers.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable[..., Any]]] = ("relu",)
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  use_bias: bool = False
  use_pre_norm: bool = False
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm:
      inputs = get_norm_layer(cfg)(
          name="mlp_layer_norm",
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
      )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = DenseGeneral(
          (len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          name="wi",
          quant=self.quant,
          use_bias=self.use_bias,
          matmul_precision=self.config.matmul_precision,
      )(inputs)
      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        x = DenseGeneral(
            self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            name=dense_name,
            quant=self.quant,
            use_bias=self.use_bias,
            matmul_precision=self.config.matmul_precision,
            dot_kernel="te_dot" if self.config.te_dense_general else None,
        )(inputs)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_mlp"))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="wo",
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
        dot_kernel="te_dot" if self.config.te_dense_general else None,
    )(x)

    output = checkpoint_name(output, "mlpwo")
    return output
