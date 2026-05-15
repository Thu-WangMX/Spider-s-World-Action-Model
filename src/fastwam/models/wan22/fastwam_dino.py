"""FastWAM-DINO: FastWAM variant that operates in DINO feature space.

This module replaces the VAE latent space with a frozen DINO encoder for visual
representation. The Video Expert predicts velocity in DINO feature space instead
of VAE latent space, while keeping the same MoT + Action Expert architecture
and KV-cache-based fast inference.

Key differences from original FastWAM:
- Uses frozen DINOv2/v3 encoder instead of frozen VAE for visual encoding
- Video Expert operates on DINO patch features (dim=1024) instead of VAE latents (dim=48)
- No video decoding capability (DINO has no decoder) — this is action-only at inference
- Training loss is flow matching MSE in DINO feature space
- Inference uses the same KV-cache trick: encode first frame with DINO, prefill video cache,
  then iteratively denoise action
"""

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .dino_encoder import DinoVideoEncoder
from .dino_video_dit import DinoVideoDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM_DINO(nn.Module):
    """FastWAM variant with DINO feature space for video representation.

    Architecture:
        - DinoVideoEncoder (frozen): video frames → DINO patch features
        - DinoVideoDiT (trainable): Video Expert in DINO space
        - ActionDiT (trainable): Action Expert (same as original FastWAM)
        - MoT (trainable): Mixed attention between video and action experts

    Training:
        - Encode GT video with frozen DINO → clean DINO features
        - Add noise via flow matching → noisy DINO features
        - MoT forward: video + action tokens joint attention
        - Loss = MSE(predicted_velocity, target_velocity) in DINO space + action space

    Inference (action-only, fast):
        - Encode first frame with DINO → clean first-frame features
        - Video Expert prefill → cache KV
        - Action denoising loop (20 steps) reusing cached video KV
    """

    def __init__(
        self,
        video_expert: DinoVideoDiT,
        action_expert: ActionDiT,
        mot: MoT,
        dino_encoder: DinoVideoEncoder,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Trainer compatibility: optimizer and freeze logic use `model.dit`
        self.dit = self.mot

        self.dino_encoder = dino_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)

        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_config(
        cls,
        dino_config: dict[str, Any],
        video_dit_config: dict[str, Any],
        action_dit_config: dict[str, Any],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        text_encoder=None,
        tokenizer=None,
        text_dim: int = 4096,
        proprio_dim: Optional[int] = None,
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        load_text_encoder: bool = False,
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_dit_init_from_wan: bool = False,
        wan_model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    ):
        """Create FastWAM_DINO from configuration dicts."""

        # Build DINO encoder (frozen)
        dino_encoder = DinoVideoEncoder(
            model_name=dino_config.get("model_name", "dinov3-vitl16"),
            model_path=dino_config.get("model_path", None),
            input_resolution=tuple(dino_config.get("input_resolution", [224, 224])),
            patch_size=dino_config.get("patch_size", 16),
            feature_dim=dino_config.get("feature_dim", 1024),
            use_cls_token=dino_config.get("use_cls_token", False),
            normalize_features=dino_config.get("normalize_features", False),
        )
        dino_encoder.load_backbone(device=torch.device(device), dtype=torch_dtype)

        # Build DINO Video Expert
        video_expert = DinoVideoDiT(
            hidden_dim=video_dit_config["hidden_dim"],
            dino_dim=dino_config.get("feature_dim", 1024),
            ffn_dim=video_dit_config["ffn_dim"],
            text_dim=video_dit_config.get("text_dim", text_dim),
            freq_dim=video_dit_config.get("freq_dim", 256),
            eps=video_dit_config.get("eps", 1e-6),
            num_heads=video_dit_config["num_heads"],
            attn_head_dim=video_dit_config["attn_head_dim"],
            num_layers=video_dit_config["num_layers"],
            video_attention_mask_mode=video_dit_config.get(
                "video_attention_mask_mode", "first_frame_causal"
            ),
            use_gradient_checkpointing=video_dit_config.get("use_gradient_checkpointing", True),
        ).to(device=device, dtype=torch_dtype)

        # Optionally initialize DinoVideoDiT from Wan2.2 Video DiT weights
        if video_dit_init_from_wan:
            from .helpers.loader import load_wan22_ti2v_5b_components

            wan_dit_config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "hidden_dim": video_dit_config["hidden_dim"],
                "ffn_dim": video_dit_config["ffn_dim"],
                "freq_dim": video_dit_config.get("freq_dim", 256),
                "text_dim": video_dit_config.get("text_dim", text_dim),
                "out_dim": 48,
                "num_heads": video_dit_config["num_heads"],
                "attn_head_dim": video_dit_config["attn_head_dim"],
                "num_layers": video_dit_config["num_layers"],
                "eps": video_dit_config.get("eps", 1e-6),
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
                "video_attention_mask_mode": "first_frame_causal",
                "action_conditioned": False,
            }
            logger.info(f"Loading Wan2.2 DiT from '{wan_model_id}' to initialize DinoVideoDiT...")
            wan_components = load_wan22_ti2v_5b_components(
                device=device,
                torch_dtype=torch_dtype,
                model_id=wan_model_id,
                tokenizer_model_id=tokenizer_model_id,
                tokenizer_max_len=tokenizer_max_len,
                redirect_common_files=True,
                dit_config=wan_dit_config,
                skip_dit_load_from_pretrain=False,
                load_text_encoder=False,
            )
            video_expert.init_from_wan_dit(wan_components.dit.state_dict())
            del wan_components  # free memory

        # Build Action Expert
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        # Validate MoT compatibility
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match DinoVideoDiT for MoT mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match DinoVideoDiT for MoT mixed attention."
            )
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("ActionDiT `num_layers` must match DinoVideoDiT.")

        # Build MoT
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )


        # Optionally load text encoder
        loaded_text_encoder = text_encoder
        loaded_tokenizer = tokenizer
        if load_text_encoder and loaded_text_encoder is None:
            # Load only text encoder + tokenizer from Wan2.2 components.
            # We must provide a valid dit_config to satisfy the loader, but
            # skip_dit_load_from_pretrain=True avoids downloading/loading the
            # actual video DiT weights (they are unused in DINO mode).
            dummy_dit_config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "hidden_dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": text_dim,
                "out_dim": 48,
                "num_heads": 24,
                "attn_head_dim": 128,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
                "video_attention_mask_mode": "first_frame_causal",
                "action_conditioned": False,
            }
            from .helpers.loader import load_wan22_ti2v_5b_components
            components = load_wan22_ti2v_5b_components(
                device=device,
                torch_dtype=torch_dtype,
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                tokenizer_model_id=tokenizer_model_id,
                tokenizer_max_len=tokenizer_max_len,
                redirect_common_files=True,
                dit_config=dummy_dit_config,
                skip_dit_load_from_pretrain=True,
                load_text_encoder=True,
            )
            loaded_text_encoder = components.text_encoder
            loaded_tokenizer = components.tokenizer


        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            dino_encoder=dino_encoder,
            text_encoder=loaded_text_encoder,
            tokenizer=loaded_tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        # dino_encoder is frozen, but still needs to be on the correct device
        if self.dino_encoder is not None and self.dino_encoder._loaded:
            self.dino_encoder.backbone.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        """Encode text prompt using text encoder."""
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append proprioception token to context sequence."""
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_dino(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Encode video frames to DINO features.

        Args:
            video_tensor: [B, 3, T, H, W] RGB video, values in [-1, 1].

        Returns:
            [B, D_dino, T, H_grid, W_grid] DINO feature latents.
        """
        return self.dino_encoder.encode_video_to_latent(video_tensor)

    @torch.no_grad()
    def _encode_single_frame_dino(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single image to DINO features.

        Args:
            image: [1, 3, H, W] or [3, H, W] single RGB image.

        Returns:
            [1, D_dino, 1, H_grid, W_grid] DINO features for one frame.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # Add temporal dimension: [B, 3, H, W] → [B, 3, 1, H, W]
        video = image.unsqueeze(2)
        return self._encode_video_dino(video)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for MoT (video + action)."""
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video → video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action → action (full visibility)
        mask[video_seq_len:, video_seq_len:] = True
        # action → first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_frame: bool,
    ) -> torch.Tensor:
        """Compute per-sample video loss in DINO feature space.

        Args:
            pred_video: [B, D, T, H, W] predicted velocity.
            target_video: [B, D, T, H, W] target velocity.
            image_is_pad: [B, T_original] padding mask for frames.
            include_initial_frame: Whether loss includes the first frame.

        Returns:
            [B] per-sample loss values.
        """
        # MSE over (D, H, W), keep temporal dimension for masking
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))  # [B, T]

        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        # For DINO, temporal_downsample_factor = 1 (no temporal compression)
        if include_initial_frame:
            video_is_pad = image_is_pad
        else:
            video_is_pad = image_is_pad[:, 1:]

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"Video loss mask shape mismatch: mask={video_is_pad.shape[1]}, "
                f"loss={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def build_inputs(self, sample: dict, tiled: bool = False) -> dict:
        """Process training sample: encode video with DINO, validate inputs.

        Args:
            sample: Dict with keys 'video' [B, 3, T, H, W], 'action' [B, T_a, a_dim],
                    'context' [B, L, D], 'context_mask' [B, L], optionally 'proprio',
                    'action_is_pad', 'image_is_pad'.

        Returns:
            Dict with processed tensors ready for training_loss.
        """
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("FastWAM_DINO training requires `context` and `context_mask`.")
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)

        if video.ndim != 5:
            raise ValueError(f"`video` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape

        if "action" not in sample:
            raise ValueError("`action` is required for training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`action` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")

        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)

        # Encode video with frozen DINO
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_dino(input_video)  # [B, D, T, H_g, W_g]

        # First frame for conditioning
        first_frame_latents = input_latents[:, :, 0:1]

        # Context
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got shapes "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # Proprio
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`proprio` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            proprio = proprio[:, 0, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def training_loss(self, sample: dict, tiled: bool = False) -> tuple[torch.Tensor, dict]:
        """Compute training loss for FastWAM-DINO.

        Flow matching in DINO feature space:
        - target = noise - clean_latent (velocity field)
        - Loss = MSE(predicted_velocity, target_velocity)

        Returns:
            (loss_total, loss_dict) where loss_dict has 'loss_video' and 'loss_action'.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]  # [B, D, T, H_g, W_g]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # === Video flow matching ===
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        # Fuse first frame (clean, no noise)
        first_frame_latents = inputs["first_frame_latents"]
        latents[:, :, 0:1] = first_frame_latents

        # === Action flow matching ===
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # === Pre-DiT for both experts ===
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,  # First frame is clean
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        # === MoT forward ===
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        # === Post-DiT ===
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # === Compute losses ===
        # Skip first frame (it's clean conditioning, not predicted)
        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_frame=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # Action loss
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)  # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Predict action velocity using cached video KV."""
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        action_horizon: int = 16,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Infer action using DINO-encoded first frame and KV-cache.

        This is the primary inference method. It:
        1. Encodes the input image with frozen DINO
        2. Runs video expert once to build KV cache
        3. Iteratively denoises action using cached video features

        Args:
            prompt: Text instruction (mutually exclusive with context/context_mask).
            input_image: [1, 3, H, W] or [3, H, W] current observation image.
            action_horizon: Number of action timesteps to predict.
            proprio: [D] or [1, D] proprioception state.
            context: [1, L, D] pre-computed text embeddings.
            context_mask: [1, L] text mask.
            num_inference_steps: Number of denoising steps.
            sigma_shift: Optional override for inference scheduler shift.
            seed: Random seed for reproducibility.
            rand_device: Device for random number generation.
            tiled: Unused (kept for API compatibility with original FastWAM).

        Returns:
            Dict with key 'action': [T_a, a_dim] predicted action chunk.
        """
        self.eval()

        if input_image is None:
            raise ValueError("`input_image` is required for inference.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1, 3, H, W] or [3, H, W], got {tuple(input_image.shape)}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # Initialize noisy action
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        # Encode input image with DINO
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_single_frame_dino(input_image)  # [1, D, 1, H_g, W_g]

        # Prepare context
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both be provided.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )

        # Video prefill: encode first frame, cache KV
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )

        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        # Action denoising loop
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )

            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save model checkpoint (only trainable components)."""
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.mot.load_state_dict(checkpoint["mot"], strict=False)
        if self.proprio_encoder is not None and "proprio_encoder" in checkpoint:
            self.proprio_encoder.load_state_dict(checkpoint["proprio_encoder"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", None)
        logger.info(f"FastWAM_DINO checkpoint loaded from {path}, step={step}")
        return step
