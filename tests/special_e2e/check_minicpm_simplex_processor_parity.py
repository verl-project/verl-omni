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
"""Compare actor/rollout/teacher tokenization using real checkpoint processor assets."""

import argparse

import numpy as np
from PIL import Image
from transformers import AutoProcessor
from vllm.config import ModelConfig
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMDummyInputsBuilder,
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMO45OmniLLMProcessingInfo,
)

from verl_omni.pipelines.minicpm.omni_rollout_adapter import MINICPM_PROMPT_KEY, MiniCPMRolloutAdapter
from verl_omni.pipelines.minicpm.processor import (
    prepare_minicpmo_inputs,
    render_minicpmo_messages,
    split_minicpmo_actor_inputs,
)


def main():
    """Check media expansion and unchanged student suffixes without loading model weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    config = ModelConfig(
        model=args.model, trust_remote_code=True, max_model_len=4096, limit_mm_per_prompt={"image": 2, "audio": 1}
    )
    info = MiniCPMO45OmniLLMProcessingInfo(InputProcessingContext(config, processor.tokenizer))
    serving_processor = MiniCPMO45OmniLLMMultiModalProcessor(info, MiniCPMO45OmniLLMDummyInputsBuilder(info))
    for image_count, audio_count in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 1)):
        images = [Image.new("RGB", (32, 32), (10, 20, 30)) for _ in range(image_count)]
        audios = [np.zeros(16000, dtype=np.float32)] * audio_count
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in images),
                    *({"type": "audio"} for _ in audios),
                    {"type": "text", "text": "Describe this."},
                ],
            }
        ]
        rendered = render_minicpmo_messages(processor, messages, add_generation_prompt=True, enable_thinking=False)
        ids, _ = split_minicpmo_actor_inputs(prepare_minicpmo_inputs(processor, rendered, images=images, audios=audios))
        media = {}
        if images:
            media["image"] = images
        if audios:
            media["audio"] = audios
        kwargs = {
            MINICPM_PROMPT_KEY: {
                "source_ids": processor.tokenizer.encode(rendered, add_special_tokens=False),
                "expanded_ids": ids,
            }
        }
        for suffix in ([], processor.tokenizer.encode("A test response.", add_special_tokens=False)):
            prompt = MiniCPMRolloutAdapter.prepare_engine_prompt(ids + suffix, None, media, kwargs)
            result = serving_processor.apply(
                ProcessorInputs(
                    prompt=prompt["prompt_token_ids"], mm_data_items=info.get_data_parser().parse_mm_data(media)
                ),
                TimingContext(enabled=False),
            )
            if result["prompt_token_ids"] != ids + suffix:
                raise AssertionError(f"Token mismatch for {image_count} images / {audio_count} audio clips.")
        print(f"PASS: {image_count} images, {audio_count} audio clips; {len(ids)} prompt tokens and unchanged response")


if __name__ == "__main__":
    main()
