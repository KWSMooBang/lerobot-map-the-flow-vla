# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from lerobot.analysis.map_the_flow import (
    AttentionKnockoutSpec,
    SpanMap,
    TokenMaskMap,
    attention_knockout_token_context,
)
from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
    from transformers.feature_extraction_utils import BatchFeature
else:
    AutoConfig = None
    AutoModel = None
    PretrainedConfig = object
    PreTrainedModel = object
    BatchFeature = None

try:
    import tree
except ImportError:
    tree = None

from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .utils import ensure_eagle_cache_ready

DEFAULT_VENDOR_EAGLE_PATH = str((Path(__file__).resolve().parent / "eagle2_hg_model").resolve())
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"


class EagleBackbone(nn.Module):
    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str = DEFAULT_VENDOR_EAGLE_PATH,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        # Prefer loading Eagle model config from the cache directory where vendor files were copied.
        vendor_dir = DEFAULT_VENDOR_EAGLE_PATH
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        try:
            ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
        except Exception as exc:  # nosec: B110
            print(f"[GROOT] Warning: failed to prepare Eagle cache for backbone: {exc}")

        config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
        # Eagle25VLForConditionalGeneration does not support flash_attention_2, and Map-the-Flow's
        # knockout intervention requires eager attention to take effect anyway (FA2/SDPA fuse the
        # mask path and ignore additive biases). Force eager up-front so construction succeeds on
        # transformers versions that strictly validate the requested implementation.
        config._attn_implementation = "eager"
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = "eager"
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = "eager"
        self.eagle_model = AutoModel.from_config(
            config,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()
        self._attention_knockout: AttentionKnockoutSpec | None = None

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.num_eagle_layers = len(self.eagle_model.language_model.model.layers)
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def set_attention_knockout(self, attention_knockout: AttentionKnockoutSpec | None) -> None:
        self._attention_knockout = attention_knockout

    @property
    def _active_knockout(self) -> AttentionKnockoutSpec | None:
        if self.training:
            return None
        return self._attention_knockout

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        token_spans, token_masks = self._build_token_layout(eagle_input)
        seq_len = eagle_input["input_ids"].shape[1]
        backbone_attention_mask = eagle_input["attention_mask"]
        if self._active_knockout is not None and "attention_mask" in eagle_input:
            eagle_input = dict(eagle_input)
            eagle_input["attention_mask"] = self._make_causal_attention_bias(eagle_input["attention_mask"])
            self._force_language_model_eager_attention()

        token_span_fallback = {} if token_masks else token_spans
        with attention_knockout_token_context(
            self.eagle_model.language_model,
            spec=self._active_knockout,
            query_length=seq_len,
            key_length=seq_len,
            source_spans=token_span_fallback,
            target_spans=token_span_fallback,
            source_token_masks=token_masks,
            target_token_masks=token_masks,
        ):
            eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, backbone_attention_mask, token_spans, token_masks

    def _force_language_model_eager_attention(self) -> None:
        for module in (
            self.eagle_model,
            getattr(self.eagle_model, "language_model", None),
            getattr(getattr(self.eagle_model, "language_model", None), "model", None),
        ):
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"

    def _make_causal_attention_bias(self, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.bool()
        batch_size, seq_len = valid.shape
        device = attention_mask.device
        dtype = next(self.eagle_model.language_model.parameters()).dtype
        if dtype == torch.float16 or dtype == torch.bfloat16:
            mask_dtype = dtype
        else:
            mask_dtype = torch.float32
        positions = torch.arange(seq_len, device=device)
        causal = positions[None, :] <= positions[:, None]
        allowed = valid[:, None, :, None] & valid[:, None, None, :] & causal[None, None, :, :]
        bias = torch.full(
            (batch_size, 1, seq_len, seq_len),
            torch.finfo(mask_dtype).min,
            dtype=mask_dtype,
            device=device,
        )
        return bias.masked_fill(allowed, 0)

    def _build_token_layout(self, eagle_input: dict) -> tuple[SpanMap, TokenMaskMap]:
        """Build token role descriptors for Map-the-Flow.

        Returns a pair ``(spans, token_masks)``:

        * ``spans`` is a fallback ``SpanMap`` (a single sample-0-derived layout). Useful
          when a downstream consumer only supports spans, and as the aggregate "prefix"
          group when batch layouts are heterogeneous.
        * ``token_masks`` is a per-batch ``TokenMaskMap`` of shape ``(B, seq_len)`` bool
          tensors. Each sample gets its own vision/language/view0/view1/... positions,
          so heterogeneous batches (e.g., different prompts) get correct per-sample
          knockout instead of sharing sample 0's layout.
        """

        input_ids = eagle_input.get("input_ids")
        attention_mask = eagle_input.get("attention_mask")
        if input_ids is None:
            return {}, {}

        if attention_mask is None:
            valid_batch = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid_batch = attention_mask.bool()

        image_token_id = getattr(self.eagle_model.config, "image_token_index", None)

        # Per-batch token masks (canonical group names match the analysis module).
        token_masks: TokenMaskMap = {
            "prefix": valid_batch.detach(),
        }
        if image_token_id is not None:
            is_image_batch = (input_ids == image_token_id) & valid_batch
            is_language_batch = valid_batch & ~is_image_batch
            token_masks["vision"] = is_image_batch.detach()
            token_masks["language"] = is_language_batch.detach()
            per_sample_view_spans = [
                self._contiguous_spans(is_image_batch[batch_idx].detach().cpu())
                for batch_idx in range(is_image_batch.shape[0])
            ]
            max_views = max((len(spans) for spans in per_sample_view_spans), default=0)
            for view_idx in range(max_views):
                view_mask = torch.zeros_like(is_image_batch)
                for batch_idx, sample_spans in enumerate(per_sample_view_spans):
                    if view_idx >= len(sample_spans):
                        continue
                    start, end = sample_spans[view_idx]
                    view_mask[batch_idx, start:end] = is_image_batch[batch_idx, start:end]
                token_masks[f"view{view_idx}"] = view_mask.detach()

        # Fallback spans from sample 0 (used by code paths that only consume SpanMap).
        ids0 = input_ids[0].detach().cpu()
        valid0 = valid_batch[0].detach().cpu()
        if image_token_id is None:
            spans: SpanMap = {"prefix": self._contiguous_spans(valid0)}
            return spans, token_masks

        is_image0 = (ids0 == image_token_id) & valid0
        image_spans0 = self._contiguous_spans(is_image0)
        spans = {
            "vision": image_spans0,
            "language": self._contiguous_spans(valid0 & ~is_image0),
            "prefix": self._contiguous_spans(valid0),
        }
        for view_idx, span in enumerate(image_spans0):
            spans[f"view{view_idx}"] = [span]
        return spans, token_masks

    def _contiguous_spans(self, mask: torch.Tensor) -> list[tuple[int, int]]:
        spans = []
        start = None
        for idx, keep in enumerate(mask.tolist()):
            if keep and start is None:
                start = idx
            elif not keep and start is not None:
                spans.append((start, idx))
                start = None
        if start is not None:
            spans.append((start, int(mask.numel())))
        return spans

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask, token_spans, token_masks = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={
                "backbone_features": eagle_embeds,
                "backbone_attention_mask": eagle_mask,
                "backbone_token_spans": token_spans,
                "backbone_token_masks": token_masks,
                "backbone_vl_layer_offset": self.num_eagle_layers,
            }
        )  # [B, T2, hidden_size]


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
class GR00TN15Config(PretrainedConfig):
    model_type = "gr00t_n1_5"

    backbone_cfg: dict
    action_head_cfg: dict
    action_horizon: int
    action_dim: int
    compute_dtype: str = "float32"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00TN15(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00TN15Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00TN15Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype
        self._attention_knockout = None
        self.post_init()

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if ACTION in inputs:
            action = inputs[ACTION]
            # In inference, action may be omitted or None; validate only when it's a tensor.
            if action is None:
                pass  # allow None during inference
            elif isinstance(action, torch.Tensor):
                shape_ok = (
                    len(action.shape) == 3
                    and action.shape[1] == self.action_horizon
                    and action.shape[2] == self.action_dim
                )
                if not shape_ok:
                    error_msg += f"\n{action.shape=}"
                    detected_error = True
            else:
                # Unexpected non-tensor type provided for action
                error_msg += f"\nInvalid type for action: {type(action)}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature) or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def set_attention_knockout(self, attention_knockout) -> None:
        self._attention_knockout = attention_knockout
        self.backbone.set_attention_knockout(attention_knockout)
        self.action_head.set_attention_knockout(attention_knockout)

    def prepare_input(self, inputs) -> tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Cast floating tensors to a memory-efficient compute dtype when requested.
            # Rationale: Upcasting backbone activations to fp32 significantly increases VRAM.
            # When compute_dtype is bfloat16, prefer bf16 for activations to match AMP behavior.
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                if getattr(self, "compute_dtype", None) == "bfloat16":
                    return x.to(self.device, dtype=torch.bfloat16)
                # Fallback: preserve previous behavior if not using bf16 compute
                return x.to(self.device, dtype=self.action_head.dtype)
            # Non-floating tensors: move device only
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model
