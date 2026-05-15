"""DINO-space Video DiT Expert for FastWAM_DINO.

This module implements a Video DiT that operates in DINO feature space instead of
VAE latent space. It shares the same MoT interface (pre_dit/post_dit) as the original
WanVideoDiT, but is adapted for DINO patch features as input/output.

Key differences from WanVideoDiT:
- in_dim = DINO feature_dim (1024 for ViT-L) instead of VAE channels (48)
- patch_size = [1, 1, 1] since DINO already produces spatial patches
- No patchify Conv3d needed; uses a linear projection instead
- No unpatchify needed; head directly outputs DINO-dim features
- Simpler spatial structure: tokens are already at patch level
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)

logger = get_logger(__name__)


class DinoHead(nn.Module):
    """Output head for DINO-space Video DiT. Maps hidden_dim back to DINO feature_dim."""

    def __init__(self, hidden_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            # Per-token modulation: t_mod shape [B, seq_len, hidden_dim]
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class DinoVideoDiT(nn.Module):
    """Video DiT Expert operating in DINO feature space.

    This module provides the same pre_dit/post_dit interface as WanVideoDiT,
    making it a drop-in replacement for the video expert in MoT.

    Architecture:
    - Input: [B, D_dino, T, H_grid, W_grid] DINO features (noisy during training)
    - Linear projection → hidden_dim tokens
    - N DiTBlocks with self-attention, cross-attention, FFN
    - DinoHead → [B, D_dino, T, H_grid, W_grid] predicted velocity

    Args:
        hidden_dim: Transformer hidden dimension (must satisfy num_heads * attn_head_dim).
        dino_dim: DINO feature dimension (e.g., 1024 for ViT-L).
        ffn_dim: Feed-forward network intermediate dimension.
        text_dim: Text encoder embedding dimension.
        freq_dim: Sinusoidal timestep embedding dimension.
        eps: LayerNorm epsilon.
        num_heads: Number of attention heads (must match ActionDiT for MoT).
        attn_head_dim: Per-head dimension (must match ActionDiT for MoT).
        num_layers: Number of DiT blocks (must match ActionDiT for MoT).
        video_attention_mask_mode: Attention mask mode for video tokens.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        hidden_dim: int,
        dino_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        video_attention_mask_mode: str = "first_frame_causal",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dino_dim = dino_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.video_attention_mask_mode = video_attention_mask_mode
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        # Input projection: DINO feature_dim → hidden_dim
        self.input_projection = nn.Linear(dino_dim, hidden_dim)

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )

        # Output head: hidden_dim → dino_dim
        self.head = DinoHead(hidden_dim, dino_dim, eps)

        # RoPE frequency tables (3D: temporal + spatial H + spatial W)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

        # For compatibility with fastwam.py first-frame fuse logic
        self.fuse_vae_embedding_in_latents = True
        self.seperated_timestep = True

        if use_gradient_checkpointing:
            logger.info("DinoVideoDiT: gradient checkpointing enabled.")

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for video self-attention within MoT."""
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    f"`video_seq_len` must be divisible by `video_tokens_per_frame`, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_frames, num_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare tokens, freqs, t_mod, context for MoT forward.

        Args:
            x: [B, D_dino, T, H_grid, W_grid] noisy DINO features (or clean first frame).
            timestep: [B] diffusion timestep.
            context: [B, L, text_dim] text embeddings.
            context_mask: [B, L] boolean mask.
            action: Not used for DINO video expert (action conditioning via cross-attn is optional).
            fuse_vae_embedding_in_latents: If True, first frame has timestep=0 (clean conditioning).

        Returns:
            Dict with keys: tokens, freqs, t_mod, context, context_mask, meta.
        """
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B, D, T, H, W], got shape {tuple(x.shape)}")

        batch_size, dino_dim, num_frames, height_grid, width_grid = x.shape
        tokens_per_frame = height_grid * width_grid

        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got shape {tuple(timestep.shape)}")

        if context_mask is None:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device
            )

        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        # Time embedding with per-token timestep (first frame = 0 if fuse mode)
        if fuse_vae_embedding_in_latents:
            token_timesteps = torch.ones(
                (batch_size, num_frames, tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0  # First frame is clean (no noise)
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep)
            t = self.time_embedding(t_emb)
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        # Reshape DINO features to token sequence: [B, D, T, H, W] → [B, T*H*W, D]
        x_tokens = rearrange(x, "b d t h w -> b (t h w) d")

        # Project from DINO dim to hidden dim
        x_tokens = self.input_projection(x_tokens)

        # Text embedding
        context_emb = self.text_embedding(context)

        # Expand context mask for cross-attention: [B, L] → [B, seq_len, L]
        seq_len = num_frames * height_grid * width_grid
        context_mask_expanded = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # RoPE frequencies
        freqs = torch.cat(
            [
                self.freqs[0][:num_frames].view(num_frames, 1, 1, -1).expand(num_frames, height_grid, width_grid, -1),
                self.freqs[1][:height_grid].view(1, height_grid, 1, -1).expand(num_frames, height_grid, width_grid, -1),
                self.freqs[2][:width_grid].view(1, 1, width_grid, -1).expand(num_frames, height_grid, width_grid, -1),
            ],
            dim=-1,
        ).reshape(num_frames * height_grid * width_grid, 1, -1).to(x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_mask_expanded,
            "meta": {
                "grid_size": (num_frames, height_grid, width_grid),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """Convert DiT output tokens back to DINO feature space.

        Args:
            x_tokens: [B, seq_len, hidden_dim] transformer output tokens.
            pre_state: Dict from pre_dit containing metadata.

        Returns:
            [B, D_dino, T, H_grid, W_grid] predicted velocity in DINO space.
        """
        num_frames, height_grid, width_grid = pre_state["meta"]["grid_size"]

        # Apply output head: [B, seq_len, hidden_dim] → [B, seq_len, dino_dim]
        x = self.head(x_tokens, pre_state["t"])

        # Reshape back to spatial format: [B, seq_len, D] → [B, D, T, H, W]
        x = rearrange(
            x, "b (t h w) d -> b d t h w", t=num_frames, h=height_grid, w=width_grid
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ) -> torch.Tensor:
        """Full forward pass (standalone, without MoT)."""
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]

        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs,
                    context_mask=context_attn_mask, self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens, context_emb, t_mod, freqs,
                    context_mask=context_attn_mask, self_attn_mask=self_attn_mask,
                )

        return self.post_dit(x_tokens, pre_state)
