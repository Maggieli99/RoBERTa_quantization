# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout

from fairseq.quantization.utils.quant_modules import *

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        quant_mode: str = 'none',
        force_dequant: str = 'none',
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        self.quant_mode = quant_mode
        self.force_dequant = force_dequant
        self.act_bit = 8
        self.fc_weight_bit = 8
        self.fc_bias_bit = 32
        self.ln_bit = 22
        self.ln_output_bit = 32

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.activation_fn_approx = IntGELU(quant_mode=self.quant_mode, force_dequant=self.force_dequant)

        self.input_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)

        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            quant_mode=quant_mode,
            force_dequant=force_dequant,
        )

        # 32bit quantization with the maximum absolute value of 2**21
        self.pre_self_attn_layer_norm_act = QuantAct(self.ln_bit, quant_mode=self.quant_mode)

        # layer norm associated with the self attention layer
        self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.self_attn_layer_norm = IntLayerNorm(self.ln_output_bit, 
                                                 quant_mode=self.quant_mode,
                                                 force_dequant=self.force_dequant)
        self.self_attn_layer_norm.set_param(self_attn_layer_norm)

        self.fc1_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.fc2_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # 32bit quantization with the maximum absolute value of 2**21
        self.pre_final_layer_norm_act = QuantAct(self.ln_bit, quant_mode=self.quant_mode)

        # layer norm associated with the position wise feed-forward NN
        final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.final_layer_norm = IntLayerNorm(self.ln_output_bit, 
                                             quant_mode=self.quant_mode,
                                             force_dequant=self.force_dequant)
        self.final_layer_norm.set_param(final_layer_norm)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        linear = QuantLinear(weight_bit=self.fc_weight_bit, bias_bit=self.fc_bias_bit, quant_mode=self.quant_mode, per_channel=True)
        linear.set_param(nn.Linear(input_dim, output_dim))
        return quant_noise(linear, q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        linear = QuantLinear(weight_bit=self.fc_weight_bit, bias_bit=self.fc_bias_bit, quant_mode=self.quant_mode, per_channel=True)
        linear.set_param(nn.Linear(input_dim, output_dim))
        return quant_noise(linear, q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        quant_mode,
        force_dequant,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            quant_mode=quant_mode,
            force_dequant=force_dequant,
            return_output_scale=True,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        x_scaling_factor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        x, x_scaling_factor = self.input_act(x, x_scaling_factor)
        residual, residual_scaling_factor = x, x_scaling_factor
        x, x_scaling_factor, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            query_scale=x_scaling_factor,
            key_scale=x_scaling_factor,
            value_scale=x_scaling_factor,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)

        # Pre LN1 activation (+ residual addition)
        x, x_scaling_factor = self.pre_self_attn_layer_norm_act(
                x, x_scaling_factor,
                identity=residual,
                identity_scaling_factor=residual_scaling_factor)

        # LN1
        x, x_scaling_factor = self.self_attn_layer_norm(
                x, x_scaling_factor)

        # Pre FC1 activation
        x, x_scaling_factor = self.fc1_act(x, x_scaling_factor)
        residual, residual_scaling_factor = x, x_scaling_factor

        # FC1
        x, x_scaling_factor = self.fc1(x, x_scaling_factor)

        x, x_scaling_factor = self.activation_fn_approx(x, x_scaling_factor)
        x = self.activation_dropout_module(x)

        # Pre FC2 activation
        x, x_scaling_factor = self.fc2_act(x, x_scaling_factor) 

        # FC2
        x, x_scaling_factor = self.fc2(x, x_scaling_factor)

        x = self.dropout_module(x)

        # Pre LN2 activation (+ residual addition)
        x, x_scaling_factor = self.pre_final_layer_norm_act(
                x, x_scaling_factor,
                identity=residual,
                identity_scaling_factor=residual_scaling_factor)

        # LN2
        x, x_scaling_factor = self.final_layer_norm(x, x_scaling_factor)

        return x, x_scaling_factor, attn
