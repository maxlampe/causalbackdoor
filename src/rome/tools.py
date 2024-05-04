"""Adapting/grafting existing code for new classes for ablations etc."""

import copy
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP


class ZeroMLP(torch.nn.Module):
    def __init__(self, const=None):
        super().__init__()
        self.const = const

    def forward(self, x):
        if self.const is None:
            return torch.zeros(x.shape).cuda()
        else:
            return self.const.cuda()


class ZeroAttn(torch.nn.Module):
    def __init__(self, const=None):
        super().__init__()
        self.const = const

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if self.const is None:
            return torch.zeros(hidden_states.shape).cuda(), (
                torch.zeros(hidden_states.shape).cuda(),
                torch.zeros(hidden_states.shape).cuda(),
            )
        else:
            return self.const.cuda(), (self.const.cuda(), self.const.cuda())


def zeroablate_modules(model, kind: str = "mlp", layer_ind: list = [0]):
    mod_copy = copy.deepcopy(model)

    if layer_ind is not None:
        rng = layer_ind
    else:
        rng = range(mod_copy.config.n_layer)
    for i in rng:
        if kind == "mlp":
            mod_copy.transformer.h[i].mlp = ZeroMLP()
        elif kind == "attn":
            mod_copy.transformer.h[i].attn = ZeroAttn()
        else:
            raise ValueError(f"Invalid 'kind' {kind}. Please use either 'mlp' or 'attn'.")


class LinearMLP(torch.nn.Module):
    def __init__(self, mat_repl=None, bias=None):
        super().__init__()
        self.mat_repl = mat_repl
        self.bias = bias

    def forward(self, x):
        if self.mat_repl is not None:
            res = torch.matmul(x, self.mat_repl.cuda())
            if self.bias is not None:
                return res.cuda() + self.bias.cuda()
            else:
                return res.cuda()
        else:
            raise ValueError("mat_repl argument must not be None.")


class LinearAttn(torch.nn.Module):
    def __init__(self, mat_repl=None):
        super().__init__()
        self.mat_repl = mat_repl

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if self.mat_repl is not None:
            res = torch.matmul(hidden_states, self.mat_repl.cuda())
            return res.cuda(), res.cuda(), res.cuda()
        else:
            raise ValueError("mat_repl argument must not be None.")


def replace_modules(
    model, mat_repl, bias=None, kind: str = "mlp", layer_ind: list = [0]
):
    mod_copy = copy.deepcopy(model)

    if layer_ind is not None:
        rng = layer_ind
    else:
        rng = range(mod_copy.config.n_layer)
    for i in rng:
        if kind == "mlp":
            mod_copy.transformer.h[i].mlp = LinearMLP(mat_repl=mat_repl, bias=bias)
        elif kind == "attn":
            mod_copy.transformer.h[i].attn = LinearAttn(mat_repl=mat_repl)
        else:
            raise ValueError(f"Invalid 'kind' {kind}. Please use either 'mlp' or 'attn'.")

    return mod_copy


def zeroresidual_modules(
    model, kind: str = "mlp", layer_ind: list = [0], cust_foo=None
):
    mod_copy = copy.deepcopy(model)

    if layer_ind is not None:
        rng = layer_ind
    else:
        rng = range(mod_copy.config.n_layer)
    for i in rng:
        if kind == "mlp":
            curr_block = OnlyMLPGPT2Block(config=mod_copy.config)
            curr_block.ln_1 = mod_copy.transformer.h[i].ln_1
            curr_block.ln_2 = mod_copy.transformer.h[i].ln_2
            curr_block.attn = mod_copy.transformer.h[i].attn
            curr_block.mlp = mod_copy.transformer.h[i].mlp
            if cust_foo is not None:
                curr_block.custom_func = cust_foo
            mod_copy.transformer.h[i] = curr_block

        elif kind == "attn":
            curr_block = OnlyATTNGPT2Block(config=mod_copy.config)
            curr_block.ln_1 = mod_copy.transformer.h[i].ln_1
            curr_block.ln_2 = mod_copy.transformer.h[i].ln_2
            curr_block.attn = mod_copy.transformer.h[i].attn
            curr_block.mlp = mod_copy.transformer.h[i].mlp
            mod_copy.transformer.h[i] = curr_block
        else:
            raise ValueError(f"Invalid 'kind' {kind}. Please use either 'mlp' or 'attn'.")

    return mod_copy


class OnlyMLPGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon
            )

        self.mlp = GPT2MLP(inner_dim, config)
        self.custom_func = None

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection

        ############################
        if self.custom_func is None:
            hidden_states = feed_forward_hidden_states
        else:
            hidden_states = self.custom_func(feed_forward_hidden_states)
        ############################

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class OnlyATTNGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon
            )

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
