# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real random tiny models: no downloaded weights, fake model classes or remote code."""

import diffusers as d
import torch
import transformers as t

from verl_omni.model_merger.fsdp_model_merger import _transformer_class

CONFIGS = {
    "QwenImagePipeline": dict(
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=1,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    ),
    "StableDiffusion3Pipeline": dict(
        sample_size=4,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        pos_embed_max_size=4,
    ),
    "FluxPipeline": dict(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=1,
        joint_attention_dim=16,
        pooled_projection_dim=16,
        axes_dims_rope=(4, 6, 6),
    ),
    "WanPipeline": dict(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
    ),
    "LTX2Pipeline": dict(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        caption_channels=16,
        vae_scale_factors=(1, 1, 1),
    ),
    "MiniMaxH3Pipeline": dict(
        in_channels=4,
        audio_in_channels=4,
        num_layers=1,
        num_refiner_layers=1,
        hidden_size=32,
        num_attention_heads=1,
        attention_head_dim=32,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=4,
    ),
    "BooguImagePipeline": dict(
        in_channels=4,
        hidden_size=16,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        multiple_of=16,
        axes_dim_rope=(4, 2, 2),
        axes_lens=(32, 16, 16),
        instruction_feature_configs={
            "instruction_feat_dim": 16,
            "reduce_type": "mean",
            "num_instruction_feat_layers": 1,
        },
        prompt_tuning_configs={"use_prompt_tuning": False},
    ),
}
CONFIGS["QwenImageEditPlusPipeline"] = CONFIGS["QwenImagePipeline"]


def tiny_transformer(architecture):
    """Construct the actual trainable class with small geometry."""
    torch.manual_seed(7)
    return _transformer_class(architecture)(**CONFIGS[architecture])


def forward_inputs(architecture):
    """Exercise each architecture's real tensor/layout API, including both AV output heads."""
    g = torch.Generator().manual_seed(11)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    common = dict(encoder_hidden_states=rand(1, 3, 16), timestep=torch.tensor([0.5]), return_dict=False)
    if architecture.startswith("QwenImage"):
        return dict(
            common,
            hidden_states=rand(1, 4, 16),
            img_shapes=[[(1, 2, 2)]],
            encoder_hidden_states_mask=torch.ones(1, 3, dtype=torch.bool),
        )
    if architecture == "StableDiffusion3Pipeline":
        return dict(common, hidden_states=rand(1, 4, 4, 4), pooled_projections=rand(1, 16))
    if architecture == "FluxPipeline":
        return dict(
            common,
            hidden_states=rand(1, 4, 16),
            pooled_projections=rand(1, 16),
            img_ids=torch.zeros(4, 3),
            txt_ids=torch.zeros(3, 3),
        )
    if architecture == "WanPipeline":
        return dict(common, hidden_states=rand(1, 4, 1, 4, 4))
    if architecture == "LTX2Pipeline":
        return dict(
            common,
            hidden_states=rand(1, 4, 4),
            audio_hidden_states=rand(1, 3, 4),
            audio_encoder_hidden_states=rand(1, 3, 16),
            audio_timestep=torch.tensor([0.5]),
            num_frames=1,
            height=2,
            width=2,
            audio_num_frames=3,
        )
    if architecture == "MiniMaxH3Pipeline":
        return dict(
            common,
            hidden_states=rand(1, 2, 16),
            audio_hidden_states=rand(1, 2, 4),
            position_ids=torch.zeros(7, 3),
            token_tags=torch.tensor([1, 1, 1, 0, 0, 2, 2]),
            timestep_indices=torch.zeros(7, dtype=torch.long),
            text_indices=torch.arange(3),
            video_indices=torch.arange(3, 5),
            audio_indices=torch.arange(5, 7),
        )
    from boogu.models.transformers.rope import BooguImageRotaryPosEmbed

    return dict(
        hidden_states=rand(1, 4, 4, 4),
        timestep=torch.tensor([0.5]),
        instruction_hidden_states=rand(1, 3, 16),
        instruction_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        freqs_cis=BooguImageRotaryPosEmbed.get_freqs_cis((4, 2, 2), (32, 16, 16), 10000),
        return_dict=False,
    )


def run_forward(model, architecture):
    """Return deterministic CPU outputs with the model's native attention implementation."""
    model.eval()
    if architecture != "BooguImagePipeline":
        model.set_attention_backend("native")
    with torch.no_grad():
        return model(**forward_inputs(architecture))


def tiny_pipeline(architecture, transformer):
    """Build complete native pipeline objects with real tiny frozen components."""
    common = dict(transformer=transformer, scheduler=d.FlowMatchEulerDiscreteScheduler())
    if architecture.startswith("QwenImage"):
        vae = d.AutoencoderKLQwenImage(
            base_dim=8,
            z_dim=4,
            dim_mult=[1],
            num_res_blocks=1,
            temperal_downsample=[],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        encoder = t.Qwen2_5_VLForConditionalGeneration(
            t.Qwen2_5_VLConfig(
                text_config=dict(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                ),
                vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16),
            )
        )
        tokenizer = t.Qwen2Tokenizer()
        kwargs = dict(common, vae=vae, text_encoder=encoder, tokenizer=tokenizer)
        if architecture == "QwenImageEditPlusPipeline":
            kwargs["processor"] = t.Qwen2VLProcessor(
                image_processor=t.Qwen2VLImageProcessor(),
                video_processor=t.Qwen2VLVideoProcessor(),
                tokenizer=tokenizer,
            )
        return getattr(d, architecture)(**kwargs)
    if architecture in {"FluxPipeline", "StableDiffusion3Pipeline"}:
        vae = d.AutoencoderKL(latent_channels=4, block_out_channels=(8,), norm_num_groups=4, sample_size=4)
        clip_config = t.CLIPTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            projection_dim=16,
        )
        t5 = t.T5EncoderModel(t.T5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2))
        if architecture == "FluxPipeline":
            return d.FluxPipeline(
                **common,
                vae=vae,
                text_encoder=t.CLIPTextModel(clip_config),
                tokenizer=t.CLIPTokenizer(),
                text_encoder_2=t5,
                tokenizer_2=t.T5Tokenizer(extra_ids=0),
            )
        return d.StableDiffusion3Pipeline(
            **common,
            vae=vae,
            text_encoder=t.CLIPTextModelWithProjection(clip_config),
            text_encoder_2=t.CLIPTextModelWithProjection(clip_config),
            text_encoder_3=t5,
            tokenizer=t.CLIPTokenizer(),
            tokenizer_2=t.CLIPTokenizer(),
            tokenizer_3=t.T5Tokenizer(extra_ids=0),
        )
    if architecture == "WanPipeline":
        vae = d.AutoencoderKLWan(
            base_dim=8,
            z_dim=4,
            dim_mult=[1],
            num_res_blocks=1,
            temperal_downsample=[],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        encoder = t.UMT5EncoderModel(
            t.UMT5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2)
        )
        return d.WanPipeline(**common, vae=vae, text_encoder=encoder, tokenizer=t.T5Tokenizer(extra_ids=0))
    if architecture == "BooguImagePipeline":
        from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
        from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )

        vae = d.AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8,),
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=4,
        )
        mllm = t.Qwen3VLForConditionalGeneration(
            t.Qwen3VLConfig(
                text_config=dict(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                ),
                vision_config=dict(
                    depth=1,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    out_hidden_size=16,
                ),
            )
        )
        processor = t.Qwen3VLProcessor(
            image_processor=t.Qwen2VLImageProcessor(),
            video_processor=t.Qwen3VLVideoProcessor(),
            tokenizer=t.Qwen2Tokenizer(),
        )
        return BooguImagePipeline(
            transformer=transformer,
            vae=vae,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            mllm=mllm,
            processor=processor,
        )
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

    vae = d.AutoencoderKLLTX2Video(
        latent_channels=4,
        block_out_channels=(8,),
        decoder_block_out_channels=(8,),
        down_block_types=("LTX2VideoDownBlock3D",),
        layers_per_block=(1, 1),
        decoder_layers_per_block=(1, 1),
        spatio_temporal_scaling=False,
        decoder_spatio_temporal_scaling=False,
        decoder_inject_noise=False,
        downsample_type=("spatial",),
        upsample_type=("spatial",),
        upsample_residual=False,
        upsample_factor=(1,),
        patch_size=1,
    )
    audio = d.AutoencoderKLLTX2Audio(base_channels=8, ch_mult=(1,), num_res_blocks=1, latent_channels=4)
    encoder = t.Gemma3ForConditionalGeneration(
        t.Gemma3Config(
            text_config=dict(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            ),
            vision_config=dict(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=2,
            ),
        )
    )
    connectors = LTX2TextConnectors(
        caption_channels=16,
        text_proj_in_factor=2,
        video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=8,
        video_connector_num_layers=1,
        video_connector_num_learnable_registers=1,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=8,
        audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=1,
    )
    vocoder = LTX2Vocoder(
        in_channels=4,
        hidden_channels=16,
        upsample_kernel_sizes=[4],
        upsample_factors=[2],
        resnet_kernel_sizes=[3],
        resnet_dilations=[[1, 3, 5]],
    )
    return d.LTX2Pipeline(
        **common,
        vae=vae,
        audio_vae=audio,
        text_encoder=encoder,
        tokenizer=t.GemmaTokenizer(),
        connectors=connectors,
        vocoder=vocoder,
    )
