import collections.abc
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...utils import ModelOutput, logging
from .configuration_dpt import DPTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DPTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "Intel/dpt-large"
_EXPECTED_OUTPUT_SHAPE = [1, 577, 1024]

@dataclass
class BaseModelOutputWithIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains intermediate activations that can be used at later stages. Useful
    in the context of Vision models.:

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_activations (`tuple(jnp.ndarray)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """

    last_hidden_state: jnp.ndarray = None
    intermediate_activations: Optional[Tuple[jnp.ndarray, ...]] = None


@dataclass
class BaseModelOutputWithPoolingAndIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states as well as intermediate
    activations that can be used by the model at later stages.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        intermediate_activations (`tuple(jnp.ndarray)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """

    last_hidden_state: jnp.ndarray = None
    pooler_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None
    intermediate_activations: Optional[Tuple[jnp.ndarray, ...]] = None


# class FlaxDPTEmbeddings(nn.Module):
#     config: DPTConfig
#     dtype: jnp.dtype = jnp.float32

#     def setup(self):
#         self.cls_token = self.param(
#             "cls_token",
#             jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
#             (1, 1, self.config.hidden_size),
#         )
#         self.patch_embeddings = FlaxDPTPatchEmbeddings(self.config, dtype=self.dtype)
#         num_patches = self.patch_embeddings.num_patches
#         self.position_embeddings = self.param(
#             "position_embeddings",
#             jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
#             (1, num_patches + 1, self.config.hidden_size),
#         )
#         self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    
#     def _resize_pos_embed(self, config, posemb, grid_size_height, grid_size_width, start_index=1):
#       posemb_tok = posemb[:, :start_index]   # for 
#       posemb_grid = posemb[0, start_index:]
#       old_grid_size = (posemb_grid.size(0) ** 0.5).astype(jnp.int64)
#     #   posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
#     #   posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
#     #   posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)
#       pass

#     def __call__(self, pixel_values, deterministic=True):
#         batch_size = pixel_values.shape[0]
#         height, width = pixel_values.shape[1], pixel_values.shape[2]
#         patch_size = self.config.patch_size
#         position_embeddings = self._resize_pos_embed(
#             self.position_embeddings, height // patch_size, width // patch_size   # ? 384//16 = 24
#         )

#         embeddings = self.patch_embeddings(pixel_values)
#         cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
#         embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        
#         # add positional encoding to each token
#         embeddings = embeddings + position_embeddings
#         embeddings = self.dropout(embeddings)
#         pass
#     pass


class FlaxDPTViTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        self.mask_token = self.param(
            "mask_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, self.config.hidden_size),
        )
        self.patch_embeddings = FlaxDPTViTPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, num_patches + 1, self.config.hidden_size),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, config, hidden_states, height, width, position_embeddings):
        num_patches = hidden_states.shape[1] - 1
        num_positions = position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return position_embeddings
        class_pos_embed = position_embeddings[:, 0]
        patch_pos_embed = position_embeddings[:, 1:]
        dim = hidden_states.shape[-1]

        h = height // config.patch_size
        w = width // config.patch_size

        patch_pos_embed = patch_pos_embed.reshape(
            (1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        )
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))
        target_dtype = patch_pos_embed.dtype
        new_height_ratio = jnp.float32(height / math.sqrt(num_positions))
        new_width_ratio = jnp.float32(width / math.sqrt(num_positions))

        scale = jnp.array([new_height_ratio, new_width_ratio], dtype=jnp.float32)
        translation = jnp.array([0.0, 0.0], dtype=jnp.float32)

        patch_pos_embed = jax.image.scale_and_translate(
            patch_pos_embed.astype(jnp.float32),
            shape=(patch_pos_embed.shape[0], patch_pos_embed.shape[1], h, w),
            spatial_dims=(2, 3),
            scale=scale,
            translation=translation,
            method="bilinier",
            antialias=False,
        )
        patch_pos_embed = patch_pos_embed.astype(target_dtype)
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 2, 3, 1)).reshape((hidden_states.shape[0], -1, dim))

        return jnp.concatenate((class_pos_embed[jnp.newaxis, :], patch_pos_embed), axis=1)

    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.dtype
        height, width = pixel_values.shape[1], pixel_values.shape[2]

        embeddings = self.patch_embeddings(pixel_values.astype(target_dtype))

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

        embeddings = embeddings + self.interpolate_pos_encoding(
            self.config, embeddings, height, width, self.position_embeddings
        )

        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


# Copied from transformers.models.dinov2.modeling_flax_dinov2.FlaxDinov2PatchEmbeddings with Dinov2Config -> DPTConfig
class FlaxDPTViTPatchEmbeddings(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    # Copied from transformers.models.vit.modeling_flax_vit.FlaxViTPatchEmbeddings.__call__
    def __call__(self, pixel_values):
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfAttention with ViTConfig->DPTConfig
class FlaxDPTViTSelfAttention(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfOutput with ViTConfig->DPTConfig
class FlaxDPTViTSelfOutput(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTAttention with ViTConfig->DPTConfig, ViTSelfAttention->FlaxDPTViTSelfAttention, ViTSelfOutput->FlaxDPTViTSelfOutput
class FlaxDPTViTAttention(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxDPTViTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxDPTViTSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs
    

# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTIntermediate with ViTConfig->DPTConfig
class FlaxDPTViTIntermediate(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTOutput with ViTConfig->DPTConfig
class FlaxDPTViTOutput(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = hidden_states + attention_output
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTLayer with with ViTConfig->DPTConfig, FlaxViTAttention->FlaxDPTViTAttention, FlaxViTIntermediate->FlaxDPTViTIntermediate, FlaxViTOutput->FlaxDPTViTOutput
class FlaxDPTViTLayer(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxDPTViTAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxDPTViTIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxDPTViTOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in DPT, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # first residual connection
        attention_output = attention_output + hidden_states

        # in DPT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(attention_output)

        hidden_states = self.intermediate(layer_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTLayerCollection with ViTConfig->DPTConfig, FlaxViTLayer->FlaxDPTViTLayer
class FlaxDPTLayerCollection(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxDPTViTLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViTConfig->DPTConfig, FlaxViTLayer->DPTViTLayer
class FlaxDPTViTEncoder(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxDPTLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    






class FlaxDPTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPTConfig
    base_model_prefix = "dpt"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: DPTConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxDPTModule(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxDPTViTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxDPTViTEncoder(self.config, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxDPTViTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)

        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.layernorm(hidden_states)
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare DPT Model transformer outputting raw hidden-states without any specific head on top.",
    DPT_START_DOCSTRING,
)
class FlaxDPTModel(FlaxDPTPreTrainedModel):
    module_class = FlaxDPTModule


FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxDPTModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("google/DPT-base-patch16-224-in21k")
    >>> model = FlaxDPTModel.from_pretrained("google/DPT-base-patch16-224-in21k")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

overwrite_call_docstring(FlaxDPTModel, FLAX_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxDPTModel, output_type=FlaxBaseModelOutputWithPooling, config_class=DPTConfig)

class FlaxDPTViTPooler(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)



class DPTReassembleStage(nn.Module):
    pass


class DPTNeck(nn.Module):
    """
    DPTNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DPT, it includes 2 stages:

    * DPTReassembleStage
    * DPTFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # postprocessing: only required in case of a non-hierarchical backbone (e.g. ViT, BEiT)
        if self.config.backbone_config is not None and self.config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = DPTReassembleStage(self.config)

        self.convs = []
        for channel in self.config.neck_hidden_sizes:
            self.convs.append(nn.Conv(features=self.config.fusion_hidden_size, kernel_size=(3, 3)))

    pass