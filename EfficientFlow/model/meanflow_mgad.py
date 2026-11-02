"""MeanFlow U-Net with an optional memory-guided action decoder.

The decoder structure follows SUREFlow's ``ActionQueryDecoder`` (self attention,
cross attention, then FFN, each with residual LayerNorm):
https://github.com/tanvirnwu/SUREFlow_IROS_2026/blob/main/SUREFlow/policy/policy.py
Read on 2026-10-30. This is an adaptation to MP1, not the original SUREFlow
architecture: observations are MP1 encoder features and both MeanFlow interval
endpoints condition the decoder. Attention is written explicitly because the
MeanFlow objective needs JVPs and differentiation through those JVPs.

Baseline modules and state-dict keys are retained. ``mgad=False`` adds no model
parameters and executes the same U-Net operations as ConditionalUnet1D_new.
"""

import torch
from torch import nn

from .diffusion.meanflow_conditional_unet1d import ConditionalUnet1D_new


class ExplicitMultiheadAttention(nn.Module):
    """Batch-first scaled dot-product attention with double-backward support."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by positive num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, memory):
        batch_size, query_length, embed_dim = query.shape
        memory_length = memory.shape[1]

        def split_heads(value, length):
            return value.reshape(batch_size, length, self.num_heads,
                                 self.head_dim).transpose(1, 2)

        q = split_heads(self.query(query), query_length)
        k = split_heads(self.key(memory), memory_length)
        v = split_heads(self.value(memory), memory_length)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** -0.5
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2)
        attended = attended.reshape(batch_size, query_length, embed_dim)
        return self.output(attended)


class MemoryGuidedActionDecoder(nn.Module):
    """One self-attention/cross-attention/FFN block; dropout is always zero."""

    def __init__(self, embed_dim, horizon, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        if horizon < 1 or int(horizon) != horizon:
            raise ValueError("horizon must be a positive integer")
        hidden_dim = int(embed_dim * mlp_ratio)
        if hidden_dim < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        self.query_embed = nn.Parameter(torch.empty(1, int(horizon), embed_dim))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        self.self_attn = ExplicitMultiheadAttention(embed_dim, num_heads)
        self.cross_attn = ExplicitMultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, action_features, memory, time_context):
        if action_features.shape[1] != self.query_embed.shape[1]:
            raise ValueError("action horizon does not match MGAD query count")
        query = action_features + self.query_embed + time_context.unsqueeze(1)
        query = self.norm1(query + self.self_attn(query, query))
        query = self.norm2(query + self.cross_attn(query, memory))
        return self.norm3(query + self.ffn(query))


class MGADConditionalUnet1D(ConditionalUnet1D_new):
    """MP1-compatible U-Net with feature access and optional MGAD refinement.

    ``forward`` preserves the original explicit ``training`` argument: True
    returns (velocity, dispersive_features), False returns only velocity.
    ``forward_with_features`` always returns those two values plus the final
    decoder features shaped [batch, horizon, down_dims[0]]. Those features are
    not detached here; a training-only statistics head must detach its input.

    MGAD expects the flattened observation condition used by MP1's FiLM path.
    It restores one memory token per observation and appends a flow-time token.
    """

    def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=3, n_groups=8, var_fc_hidden_dim=512,
                 condition_type="film", use_down_condition=True,
                 use_mid_condition=True, use_up_condition=True, *, horizon=4,
                 mgad=True, mgad_num_obs_steps=2, mgad_num_heads=4,
                 mgad_mlp_ratio=2.0):
        super().__init__(
            input_dim=input_dim, local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims, kernel_size=kernel_size, n_groups=n_groups,
            var_fc_hidden_dim=var_fc_hidden_dim, condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition)
        self.mgad_enabled = bool(mgad)
        self.decoder_feature_dim = down_dims[0]
        self.mgad_num_obs_steps = mgad_num_obs_steps
        self.mgad_global_cond_dim = global_cond_dim
        if self.mgad_enabled:
            if (isinstance(mgad_num_obs_steps, bool)
                    or not isinstance(mgad_num_obs_steps, int)
                    or mgad_num_obs_steps < 1):
                raise ValueError("mgad_num_obs_steps must be a positive integer")
            if (global_cond_dim is None or global_cond_dim < 1
                    or global_cond_dim % mgad_num_obs_steps):
                raise ValueError("global_cond_dim must contain whole observation tokens")
            if "cross_attention" in condition_type:
                raise ValueError("MGAD expects MP1's flattened global conditioning")
            self.mgad_observation_projection = nn.Linear(
                global_cond_dim // mgad_num_obs_steps, self.decoder_feature_dim)
            self.mgad_memory_positions = nn.Parameter(torch.empty(
                1, mgad_num_obs_steps, self.decoder_feature_dim))
            nn.init.normal_(self.mgad_memory_positions, mean=0.0, std=0.02)
            self.mgad_time_projection = nn.Linear(
                diffusion_step_embed_dim * 2, self.decoder_feature_dim)
            self.mgad = MemoryGuidedActionDecoder(
                self.decoder_feature_dim, horizon,
                num_heads=mgad_num_heads, mlp_ratio=mgad_mlp_ratio)

    @staticmethod
    def _batch_time(value, sample):
        # Keep the baseline's scalar conversion semantics for disabled parity.
        if not torch.is_tensor(value):
            value = torch.tensor([value], dtype=torch.long, device=sample.device)
        elif value.ndim == 0:
            value = value[None].to(sample.device)
        return value.expand(sample.shape[0]).to(sample.device)

    def _memory(self, global_cond, t_embedding, r_embedding):
        if (global_cond is None or global_cond.ndim != 2
                or global_cond.shape[-1] != self.mgad_global_cond_dim):
            raise ValueError("MGAD requires flattened [batch, global_cond_dim] observations")
        observations = global_cond.reshape(global_cond.shape[0],
                                           self.mgad_num_obs_steps, -1)
        memory = self.mgad_observation_projection(observations)
        memory = memory + self.mgad_memory_positions
        time_context = self.mgad_time_projection(
            torch.cat((t_embedding, r_embedding), dim=-1))
        return torch.cat((memory, time_context.unsqueeze(1)), dim=1), time_context

    def forward_with_features(self, sample, timestep, local_cond=None,
                              global_cond=None, r=None, **kwargs):
        sample = sample.permute(0, 2, 1)
        t_embedding = self.diffusion_step_encoder(self._batch_time(timestep, sample))
        r_embedding = self.diffusion_step_encoder_rs(self._batch_time(r, sample))
        timestep_embed = t_embedding + r_embedding
        global_feature = timestep_embed
        if global_cond is not None:
            if self.condition_type == "cross_attention":
                timestep_embed = timestep_embed.unsqueeze(1).expand(
                    -1, global_cond.shape[1], -1)
            global_feature = torch.cat((timestep_embed, global_cond), dim=-1)

        h_local = []
        if local_cond is not None:
            local_cond = local_cond.permute(0, 2, 1)
            resnet, resnet2 = self.local_cond_encoder
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        x = sample
        skips = []
        down_latents = []
        for index, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            condition = global_feature if self.use_down_condition else None
            x = resnet(x, condition)
            if index == 0 and h_local:
                x = x + h_local[0]
            x = resnet2(x, condition)
            skips.append(x)
            down_latents.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature if self.use_mid_condition else None)

        for index, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, skips.pop()), dim=1)
            condition = global_feature if self.use_up_condition else None
            x = resnet(x, condition)
            # Preserve the original U-Net's local-conditioning branch exactly.
            if index == len(self.up_modules) and h_local:
                x = x + h_local[1]
            x = resnet2(x, condition)
            x = upsample(x)

        decoder_features = self.final_conv[0](x).permute(0, 2, 1)
        if self.mgad_enabled:
            memory, time_context = self._memory(global_cond, t_embedding, r_embedding)
            decoder_features = self.mgad(decoder_features, memory, time_context)
        velocity = self.final_conv[1](decoder_features.permute(0, 2, 1))
        velocity = velocity.permute(0, 2, 1)
        dispersive_features = torch.stack(
            [feature.flatten(1) for feature in down_latents[:-1]], dim=0)
        return velocity, dispersive_features, decoder_features

    def forward(self, sample, timestep, local_cond=None, global_cond=None,
                r=None, training=True, **kwargs):
        velocity, dispersive_features, _ = self.forward_with_features(
            sample, timestep, local_cond=local_cond, global_cond=global_cond, r=r,
            **kwargs)
        if training:
            return velocity, dispersive_features
        return velocity
