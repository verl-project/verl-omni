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
"""CPU unit tests for multimodal SDPO / OPSD on-policy self-distillation."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, NonTensorStack, TensorDict
from transfer_queue import KVBatchMeta
from verl.trainer.distillation import distillation_loss
from verl.trainer.ppo.utils import Role
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

from verl_omni.trainer.distillation.losses import omni_distillation_ppo_loss
from verl_omni.trainer.distillation.self_distillation import (
    EMATeacherState,
    align_teacher_sequence_outputs,
    build_multimodal_teacher_prompt,
    build_neutral_teacher_outputs,
    extract_sequence_rewards,
    reprompt_multimodal_messages,
    select_teacher_contexts,
    summarize_self_distillation_metrics,
)
from verl_omni.trainer.main_omni import _build_self_distillation_task_runner_v1
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl_omni.trainer.omni.ray_omni_trainer_separate_async import OmniPPOTrainerSeparateAsync
from verl_omni.workers.config.omni.distillation import (
    OmniDistillationConfig,
    OmniDistillationLossConfig,
    SelfDistillationConfig,
)
from verl_omni.workers.engine_workers import ActorRolloutRefWorker


class _DummyTokenizer:
    """Whitespace tokenizer with deterministic token IDs including media placeholders."""

    pad_token_id = 0
    eos_token_id = 2
    image_token_id = 100
    audio_token_id = 101
    video_token_id = 102

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
            "<eos>": 2,
            "<image>": 100,
            "<audio>": 101,
            "<video>": 102,
        }
        self._inv: dict[int, str] = {v: k for k, v in self._vocab.items()}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            idx = max(self._vocab.values()) + 1
            self._vocab[tok] = idx
            self._inv[idx] = tok
        return self._vocab[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._id(tok) for tok in text.split() if tok]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        toks = []
        for idx in ids:
            tok = self._inv.get(int(idx), f"tok_{idx}")
            if skip_special_tokens and tok in {"<pad>", "<eos>"}:
                continue
            toks.append(tok)
        return " ".join(toks)

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ):
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<{msg['role']}>")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        parts.append("<image>")
                    elif item.get("type") == "audio":
                        parts.append("<audio>")
                    elif item.get("type") == "video":
                        parts.append("<video>")
                    elif item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = " ".join(p for p in parts if p)
        return self.encode(rendered) if tokenize else rendered


class _DummyOmniProcessor:
    """Minimal multimodal processor mirroring HF/Qwen-Omni placeholder expansion."""

    image_token_id = 100
    audio_token_id = 101
    video_token_id = 102

    def __init__(self, tokenizer: _DummyTokenizer, *, drop_media_placeholder: bool = False) -> None:
        self.tokenizer = tokenizer
        self.drop_media_placeholder = drop_media_placeholder

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ):
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.drop_media_placeholder:
            rendered = rendered.replace("<image>", "", 1)
        return self.tokenizer.encode(rendered) if tokenize else rendered

    def __call__(self, text, images=None, videos=None, audio=None, return_tensors="pt", **kwargs):
        del images, videos, audio, return_tensors, kwargs
        prompt_text = text[0] if isinstance(text, list) else text
        ids = self.tokenizer.encode(prompt_text)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class TestPrivilegedContextSelection:
    def test_sdpo_selects_winning_sibling_and_respects_self_exclusion(self):
        cfg = SelfDistillationConfig(
            enabled=True,
            mode="sdpo",
            reward_threshold=0.5,
            dont_reprompt_on_self_success=True,
        )
        decisions = select_teacher_contexts(
            uids=["g1", "g1", "g1", "g2", "g2"],
            rewards=[0.0, 1.0, 0.2, 0.0, 0.0],
            response_texts=["wrong_a", "right_b", "wrong_c", "fail_d", "fail_e"],
            config=cfg,
        )
        # g1: sample 0 and 2 use winner sample 1; sample 1 excludes itself -> fallback.
        assert decisions[0].has_context is True
        assert decisions[0].source == "sibling"
        assert decisions[0].demonstration == "right_b"
        assert decisions[0].sibling_index == 1

        assert decisions[1].has_context is False
        assert decisions[1].source == "none"

        assert decisions[2].has_context is True
        assert decisions[2].demonstration == "right_b"

        # g2: all-fail group with no feedback -> fallback.
        assert decisions[3].has_context is False
        assert decisions[4].has_context is False

    def test_sdpo_allows_self_reprompt_when_disabled_and_picks_highest_reward(self):
        cfg = SelfDistillationConfig(
            enabled=True,
            mode="sdpo",
            reward_threshold=0.5,
            dont_reprompt_on_self_success=False,
        )
        decisions = select_teacher_contexts(
            uids=["g1", "g1", "g1"],
            rewards=[0.6, 0.95, 0.0],
            response_texts=["ok_sol", "best_sol", "bad_sol"],
            config=cfg,
        )
        assert [d.sibling_index for d in decisions] == [1, 1, 1]
        assert all(d.demonstration == "best_sol" for d in decisions)

    def test_sdpo_with_verifier_feedback_without_winning_sibling(self):
        cfg = SelfDistillationConfig(enabled=True, mode="sdpo", include_environment_feedback=True)
        decisions = select_teacher_contexts(
            uids=["g1", "g1"],
            rewards=[0.0, 0.0],
            response_texts=["ans_1", "ans_2"],
            config=cfg,
            reward_extra_infos=[{"feedback": "Unit mismatch: expected meters."}, {}],
        )
        assert decisions[0].has_context is True
        assert decisions[0].source == "feedback"
        assert decisions[0].feedback == "Unit mismatch: expected meters."
        assert decisions[1].has_context is False

    def test_opsd_works_with_rollout_n_equals_1(self):
        cfg = SelfDistillationConfig(enabled=True, mode="opsd")
        decisions = select_teacher_contexts(
            uids=["u1", "u2"],
            rewards=[0.0, 0.0],
            response_texts=["guess_1", "guess_2"],
            config=cfg,
            reward_models=[{"ground_truth": "42"}, {}],
            extra_infos=[{}, {"ground_truth": "audio shows siren"}],
        )
        assert decisions[0].has_context is True
        assert decisions[0].source == "reference"
        assert decisions[0].reference_solution == "42"
        assert decisions[1].has_context is True
        assert decisions[1].reference_solution == "audio shows siren"

    def test_auto_mode_prefers_sdpo_then_falls_back_to_opsd(self):
        cfg = SelfDistillationConfig(enabled=True, mode="auto")
        decisions = select_teacher_contexts(
            uids=["g1", "g1", "g2"],
            rewards=[0.0, 1.0, 0.0],
            response_texts=["wrong", "winner", "only_wrong"],
            config=cfg,
            reward_models=[{"ground_truth": "ref_1"}, {"ground_truth": "ref_1"}, {"ground_truth": "ref_2"}],
        )
        assert decisions[0].source == "sibling"
        assert decisions[0].demonstration == "winner"
        # Winner sample 1 has no sibling winner, so auto falls back to OPSD reference.
        assert decisions[1].source == "reference"
        assert decisions[1].reference_solution == "ref_1"
        assert decisions[2].source == "reference"
        assert decisions[2].reference_solution == "ref_2"

    def test_extract_sequence_rewards_padded_and_nested(self):
        padded = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.25, 0.0]])
        assert extract_sequence_rewards(padded) == pytest.approx([1.0, 0.25])

        nested_rm = torch.nested.as_nested_tensor([torch.tensor([0.0, 1.0]), torch.tensor([0.5])], layout=torch.jagged)
        assert extract_sequence_rewards(nested_rm) == pytest.approx([1.0, 0.5])


class TestMultimodalReprompting:
    def test_preserves_image_audio_video_blocks_and_processor_tensors(self):
        tok = _DummyTokenizer()
        proc = _DummyOmniProcessor(tok)
        raw_prompt = [
            {"role": "system", "content": "You are an omni assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img_obj_0"},
                    {"type": "audio", "audio": "aud_obj_0"},
                    {"type": "video", "video": "vid_obj_0"},
                    {"type": "text", "text": "Solve the diagram and sound clip."},
                ],
            },
        ]
        orig_snapshot = copy.deepcopy(raw_prompt)
        student_prompt_ids = tok.apply_chat_template(raw_prompt, tokenize=True, add_generation_prompt=True)
        mm_data = {"images": ["img_obj_0"], "audios": ["aud_obj_0"], "videos": ["vid_obj_0"]}
        mm_inputs = {
            "pixel_values": torch.randn(4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "input_features": torch.randn(2, 8),
        }
        cfg = SelfDistillationConfig(enabled=True, mode="sdpo")

        reprompted = build_multimodal_teacher_prompt(
            raw_prompt=raw_prompt,
            student_prompt_ids=student_prompt_ids,
            privileged_text="Correct example solution:\nCompute angle = 60 degrees.",
            tokenizer=tok,
            processor=proc,
            multi_modal_data=mm_data,
            existing_multi_modal_inputs=mm_inputs,
            config=cfg,
        )

        # Caller input must not be mutated.
        assert raw_prompt == orig_snapshot

        # Exact multimodal blocks preserved in order.
        user_content = reprompted.teacher_messages[1]["content"]
        assert [item["type"] for item in user_content[:3]] == ["image", "audio", "video"]
        assert user_content[0]["image"] == "img_obj_0"
        assert user_content[1]["audio"] == "aud_obj_0"
        assert user_content[2]["video"] == "vid_obj_0"

        # Media tokens appear with identical counts in teacher_prompt_ids.
        assert reprompted.teacher_prompt_ids.count(tok.image_token_id) == 1
        assert reprompted.teacher_prompt_ids.count(tok.audio_token_id) == 1
        assert reprompted.teacher_prompt_ids.count(tok.video_token_id) == 1

        # Raw media and processor tensors preserved untouched.
        assert reprompted.multi_modal_data == mm_data
        assert reprompted.multi_modal_inputs == mm_inputs
        assert reprompted.added_prompt_tokens > 0

    def test_max_reprompt_tokens_trims_privileged_text_without_dropping_media(self):
        tok = _DummyTokenizer()
        proc = _DummyOmniProcessor(tok)
        raw_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img_0"},
                    {"type": "text", "text": "What is shown?"},
                ],
            }
        ]
        student_prompt_ids = tok.apply_chat_template(raw_prompt, tokenize=True, add_generation_prompt=True)
        long_demo = " ".join(f"word{i}" for i in range(120))
        cfg = SelfDistillationConfig(enabled=True, mode="sdpo", max_reprompt_tokens=45)

        reprompted = build_multimodal_teacher_prompt(
            raw_prompt=raw_prompt,
            student_prompt_ids=student_prompt_ids,
            privileged_text=long_demo,
            tokenizer=tok,
            processor=proc,
            multi_modal_data={"images": ["img_0"]},
            config=cfg,
        )
        assert len(reprompted.teacher_prompt_ids) <= 45
        assert reprompted.trimmed_context is True
        assert reprompted.teacher_prompt_ids.count(tok.image_token_id) == 1

    def test_placeholder_drift_raises_loud_value_error(self):
        tok = _DummyTokenizer()
        broken_proc = _DummyOmniProcessor(tok, drop_media_placeholder=True)
        raw_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img_0"},
                    {"type": "text", "text": "Count objects."},
                ],
            }
        ]
        student_prompt_ids = tok.apply_chat_template(raw_prompt, tokenize=True, add_generation_prompt=True)
        cfg = SelfDistillationConfig(enabled=True, mode="opsd", preserve_multimodal_inputs=True)

        with pytest.raises(ValueError, match="altered multimodal placeholder token counts"):
            build_multimodal_teacher_prompt(
                raw_prompt=raw_prompt,
                student_prompt_ids=student_prompt_ids,
                privileged_text="Reference solution:\n3",
                tokenizer=tok,
                processor=broken_proc,
                multi_modal_data={"images": ["img_0"]},
                config=cfg,
            )

    def test_reprompt_string_prompt_fallback(self):
        cfg = SelfDistillationConfig(enabled=True, mode="opsd")
        out = reprompt_multimodal_messages("Solve 2+2.", "Reference: 4", reprompt_template=cfg.reprompt_template)
        assert isinstance(out, str)
        assert out.startswith("Solve 2+2.")
        assert "Reference: 4" in out


class TestTeacherSequenceAlignment:
    @pytest.mark.parametrize(
        ("teacher_prompt_len", "student_prompt_len", "response_len", "width"),
        [
            (12, 5, 4, 1),
            (4, 9, 4, 1),
            (6, 6, 5, 4),
            (15, 7, 3, 4),
        ],
    )
    def test_alignment_matches_verl_no_padding_2_padding(
        self,
        teacher_prompt_len: int,
        student_prompt_len: int,
        response_len: int,
        width: int,
    ):
        teacher_seq_len = teacher_prompt_len + response_len
        student_seq_len = student_prompt_len + response_len

        positions = torch.arange(teacher_seq_len, dtype=torch.int32).unsqueeze(-1)
        teacher_ids = positions * 10 + torch.arange(width, dtype=torch.int32).unsqueeze(0)
        teacher_lps = -(positions.float() + 0.1 * torch.arange(width, dtype=torch.float32).unsqueeze(0))

        aligned_ids, aligned_lps = align_teacher_sequence_outputs(
            teacher_ids=teacher_ids,
            teacher_logprobs=teacher_lps,
            teacher_prompt_len=teacher_prompt_len,
            student_prompt_len=student_prompt_len,
            response_len=response_len,
            pad_token_id=0,
        )
        assert aligned_ids.shape == (student_seq_len, width)
        assert aligned_lps.shape == (student_seq_len, width)

        data = TensorDict(
            {
                "prompts": torch.nested.as_nested_tensor(
                    [torch.ones(student_prompt_len, dtype=torch.long)], layout=torch.jagged
                ),
                "responses": torch.nested.as_nested_tensor(
                    [torch.ones(response_len, dtype=torch.long)], layout=torch.jagged
                ),
                "teacher_ids": torch.nested.as_nested_tensor([aligned_ids], layout=torch.jagged),
                "teacher_logprobs": torch.nested.as_nested_tensor([aligned_lps], layout=torch.jagged),
            },
            batch_size=[1],
        )
        extracted_ids = no_padding_2_padding(data["teacher_ids"], data)[0]
        extracted_lps = no_padding_2_padding(data["teacher_logprobs"], data)[0]

        expected_start = teacher_prompt_len - 1
        expected_ids = teacher_ids[expected_start : expected_start + response_len]
        expected_lps = teacher_lps[expected_start : expected_start + response_len]
        torch.testing.assert_close(extracted_ids, expected_ids)
        torch.testing.assert_close(extracted_lps, expected_lps)

    def test_alignment_accepts_response_only_teacher_tensor(self):
        resp_ids = torch.tensor([[11, 12], [21, 22], [31, 32]], dtype=torch.int32)
        resp_lps = torch.tensor([[-0.2, -1.2], [-0.3, -1.3], [-0.4, -1.4]], dtype=torch.float32)
        aligned_ids, aligned_lps = align_teacher_sequence_outputs(
            teacher_ids=resp_ids,
            teacher_logprobs=resp_lps,
            teacher_prompt_len=10,
            student_prompt_len=4,
            response_len=3,
            pad_token_id=0,
        )
        assert aligned_ids.shape == (7, 2)
        torch.testing.assert_close(aligned_ids[3:6], resp_ids)
        torch.testing.assert_close(aligned_lps[3:6], resp_lps)

    def test_neutral_teacher_outputs_shape_and_values(self):
        ids, lps = build_neutral_teacher_outputs(student_prompt_len=5, response_len=3, topk=4, pad_token_id=7)
        assert ids.shape == (8, 4)
        assert lps.shape == (8, 4)
        assert (ids == 7).all()
        assert (lps == 0.0).all()


class TestSelfDistillationMixedFallbackLoss:
    @staticmethod
    def _make_actor_config() -> ActorConfig:
        return ActorConfig(
            strategy="fsdp2",
            rollout_n=2,
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=2,
            loss_agg_mode="token-mean",
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            entropy_coeff=0.0,
            use_kl_loss=False,
        )

    @staticmethod
    def _make_self_distill_config(*, fallback_to_pg: bool = True, use_task_rewards: bool = False):
        return OmniDistillationConfig(
            enabled=True,
            nnodes=0,
            n_gpus_per_node=0,
            distillation_loss=OmniDistillationLossConfig(
                loss_mode="k1",
                use_policy_gradient=True,
                use_task_rewards=use_task_rewards,
            ),
            self_teacher=SelfDistillationConfig(
                enabled=True,
                mode="sdpo",
                fallback_to_pg=fallback_to_pg,
            ),
        )

    @staticmethod
    def _make_batch(
        self_distill_mask: list[bool],
        *,
        adv_scale_0: float = 1.0,
        teacher_lp_1: float = -2.0,
    ):
        prompt_lens = [3, 4]
        resp_lens = [3, 2]
        seq_lens = [p + r for p, r in zip(prompt_lens, resp_lens, strict=True)]

        raw_log_probs_0 = torch.tensor([-0.5, -0.4, -0.8, -0.6, -0.7, -0.2], requires_grad=True)
        raw_log_probs_1 = torch.tensor([-0.3, -0.2, -0.1, -0.9, -0.5, -0.4], requires_grad=True)
        log_probs_nested = torch.nested.as_nested_tensor([raw_log_probs_0, raw_log_probs_1], layout=torch.jagged)

        old_lp_0 = torch.tensor([-0.7, -0.5, -0.6])
        old_lp_1 = torch.tensor([-0.8, -0.4])
        adv_0 = torch.tensor([1.0, 0.5, -0.5]) * adv_scale_0
        adv_1 = torch.tensor([1.2, -0.8])

        teacher_lp_0_full = torch.zeros((seq_lens[0], 1), dtype=torch.float32)
        teacher_lp_0_full[prompt_lens[0] - 1 : prompt_lens[0] - 1 + resp_lens[0], 0] = torch.tensor([-0.3, -0.2, -0.4])
        teacher_lp_1_full = torch.zeros((seq_lens[1], 1), dtype=torch.float32)
        teacher_lp_1_full[prompt_lens[1] - 1 : prompt_lens[1] - 1 + resp_lens[1], 0] = teacher_lp_1

        data = TensorDict(
            {
                "prompts": torch.nested.as_nested_tensor(
                    [torch.ones(p, dtype=torch.long) for p in prompt_lens], layout=torch.jagged
                ),
                "responses": torch.nested.as_nested_tensor(
                    [torch.ones(r, dtype=torch.long) for r in resp_lens], layout=torch.jagged
                ),
                "response_mask": torch.nested.as_nested_tensor(
                    [torch.ones(r, dtype=torch.long) for r in resp_lens], layout=torch.jagged
                ),
                "old_log_probs": torch.nested.as_nested_tensor([old_lp_0, old_lp_1], layout=torch.jagged),
                "advantages": torch.nested.as_nested_tensor([adv_0, adv_1], layout=torch.jagged),
                "teacher_ids": torch.nested.as_nested_tensor(
                    [torch.ones((s, 1), dtype=torch.int32) for s in seq_lens], layout=torch.jagged
                ),
                "teacher_logprobs": torch.nested.as_nested_tensor(
                    [teacher_lp_0_full, teacher_lp_1_full], layout=torch.jagged
                ),
                "self_distill_mask": torch.tensor(self_distill_mask, dtype=torch.bool),
                "dp_size": NonTensorData(1),
                "batch_num_tokens": NonTensorData(None),
                "global_batch_size": NonTensorData(None),
            },
            batch_size=[2],
        )
        return data, {"log_probs": log_probs_nested}, (raw_log_probs_0, raw_log_probs_1)

    def test_mixed_batch_routes_distilled_and_fallback_samples_independently(self):
        actor_cfg = self._make_actor_config()
        distill_cfg = self._make_self_distill_config(fallback_to_pg=True, use_task_rewards=False)

        data_a, out_a, _ = self._make_batch([True, False], adv_scale_0=1.0, teacher_lp_1=-2.0)
        loss_a, metrics_a = omni_distillation_ppo_loss(actor_cfg, distill_cfg, model_output=out_a, data=data_a)
        assert torch.isfinite(loss_a)
        assert metrics_a["self_distillation/distilled_tokens"].aggregate() == 3.0
        assert metrics_a["self_distillation/fallback_tokens"].aggregate() == 2.0

        # Changing sample 0's advantage or sample 1's teacher_logprobs must NOT change the loss.
        data_b, out_b, _ = self._make_batch([True, False], adv_scale_0=99.0, teacher_lp_1=-99.0)
        loss_b, _ = omni_distillation_ppo_loss(actor_cfg, distill_cfg, model_output=out_b, data=data_b)
        torch.testing.assert_close(loss_a, loss_b)

        loss_a.backward()

    def test_all_fallback_matches_ppo_loss_and_no_nans(self):
        actor_cfg = self._make_actor_config()
        distill_cfg = self._make_self_distill_config(fallback_to_pg=True)
        data, out, _ = self._make_batch([False, False])
        loss, metrics = omni_distillation_ppo_loss(actor_cfg, distill_cfg, model_output=out, data=data)
        expected_pg, _ = ppo_loss(actor_cfg, out, data)
        torch.testing.assert_close(loss, expected_pg)
        assert metrics["self_distillation/distilled_tokens"].aggregate() == 0.0
        assert metrics["self_distillation/fallback_tokens"].aggregate() == 5.0

    def test_all_distilled_matches_distillation_loss(self):
        actor_cfg = self._make_actor_config()
        distill_cfg = self._make_self_distill_config(fallback_to_pg=True)
        data, out, _ = self._make_batch([True, True])
        loss, metrics = omni_distillation_ppo_loss(actor_cfg, distill_cfg, model_output=out, data=data)
        expected_kd, _ = distillation_loss(actor_cfg, distill_cfg, out, data)
        torch.testing.assert_close(loss, expected_kd)
        assert metrics["self_distillation/distilled_tokens"].aggregate() == 5.0
        assert metrics["self_distillation/fallback_tokens"].aggregate() == 0.0

    def test_all_fallback_with_fallback_disabled_returns_differentiable_zero(self):
        actor_cfg = self._make_actor_config()
        distill_cfg = self._make_self_distill_config(fallback_to_pg=False)
        data, out, (lp0, lp1) = self._make_batch([False, False])
        loss, _ = omni_distillation_ppo_loss(actor_cfg, distill_cfg, model_output=out, data=data)
        assert loss.item() == 0.0
        assert loss.requires_grad is True
        loss.backward()
        assert torch.isfinite(lp0.grad).all()
        assert torch.isfinite(lp1.grad).all()


class TestEMATeacherState:
    def test_update_and_swap_restores_weights_including_on_exception(self):
        model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema = EMATeacherState(decay=0.9)
        ema.initialize(model)

        with torch.no_grad():
            model.weight.fill_(3.0)
        ema.update(model)
        # 0.9 * 1.0 + 0.1 * 3.0 = 1.2
        torch.testing.assert_close(ema.shadow["weight"], torch.full((2, 3), 1.2))
        assert ema.num_updates == 1

        with ema.swap_into(model):
            torch.testing.assert_close(model.weight, torch.full((2, 3), 1.2))
        torch.testing.assert_close(model.weight, torch.full((2, 3), 3.0))

        with pytest.raises(RuntimeError, match="boom"):
            with ema.swap_into(model):
                torch.testing.assert_close(model.weight, torch.full((2, 3), 1.2))
                raise RuntimeError("boom")
        torch.testing.assert_close(model.weight, torch.full((2, 3), 3.0))


class TestSelfDistillationTrainerWiring:
    def test_resource_pool_mgr_skips_teacher_pool_for_self_distillation(self):
        trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"lora_rank": 0, "lora_adapter_path": None},
                    "actor": {"use_kl_loss": False},
                },
                "algorithm": {"use_kl_in_reward": False, "adv_estimator": "grpo"},
                "critic": {"enable": False},
                "reward": {
                    "reward_model": {
                        "enable": False,
                        "enable_resource_pool": False,
                        "nnodes": 0,
                        "n_gpus_per_node": 0,
                    }
                },
                "trainer": {"nnodes": 1, "n_gpus_per_node": 4},
                "distillation": {
                    "enabled": True,
                    "nnodes": 0,
                    "n_gpus_per_node": 0,
                    "self_teacher": {"enabled": True, "mode": "sdpo"},
                },
            }
        )
        trainer._init_resource_pool_mgr()

        assert Role.TeacherModel not in trainer.mapping
        assert "teacher_pool" not in trainer.resource_pool_manager.resource_pool_spec
        assert trainer.resource_pool_manager.resource_pool_spec["global_pool"] == [4]
        assert trainer.config.distillation.enabled is True

        trainer.teacher_model_manager = None
        assert trainer.get_teacher_client() is None

    def test_compute_self_teacher_log_probs_scores_multimodal_and_writes_tq(self):
        tok = _DummyTokenizer()
        proc = _DummyOmniProcessor(tok)

        trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
        trainer.tokenizer = tok
        trainer.processor = proc
        trainer.config = OmegaConf.create({"data": {"mm_processor_kwargs": {"sampling_rate": 16000}}})
        trainer.distillation_config = OmniDistillationConfig(
            enabled=True,
            nnodes=0,
            n_gpus_per_node=0,
            distillation_loss=OmniDistillationLossConfig(
                loss_mode="forward_kl_topk",
                topk=2,
                use_policy_gradient=False,
            ),
            self_teacher=SelfDistillationConfig(enabled=True, mode="sdpo", reward_threshold=0.5),
        )

        # 3 samples: g1 has loser (idx 0) + winner (idx 1); g2 has single loser (idx 2, fallback).
        p0 = torch.tensor(tok.encode("<user> <image> <audio> q1 <assistant>"), dtype=torch.long)
        p1 = p0.clone()
        p2 = torch.tensor(tok.encode("<user> q2 <assistant>"), dtype=torch.long)
        r0 = torch.tensor(tok.encode("wrong answer"), dtype=torch.long)
        r1 = torch.tensor(tok.encode("right answer"), dtype=torch.long)
        r2 = torch.tensor(tok.encode("all fail"), dtype=torch.long)

        raw_mm_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "pil_img_0"},
                    {"type": "audio", "audio": "wav_aud_0"},
                    {"type": "text", "text": "q1"},
                ],
            }
        ]
        raw_text_prompt = [{"role": "user", "content": "q2"}]

        tq_data = TensorDict(
            {
                "prompts": torch.nested.as_nested_tensor([p0, p1, p2], layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor([r0, r1, r2], layout=torch.jagged),
                "response_mask": torch.nested.as_nested_tensor(
                    [torch.ones_like(r0), torch.ones_like(r1), torch.ones_like(r2)], layout=torch.jagged
                ),
                "rm_scores": torch.nested.as_nested_tensor(
                    [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])],
                    layout=torch.jagged,
                ),
                "uid": NonTensorStack("g1", "g1", "g2"),
                "raw_prompt": NonTensorStack(raw_mm_prompt, raw_mm_prompt, raw_text_prompt),
            },
            batch_size=[3],
        )

        generate_calls: list[dict] = []

        class _FakeLLMClient:
            async def generate(self, **kwargs):
                generate_calls.append(kwargs)
                seq_len = len(kwargs["prompt_ids"])
                topk = kwargs["sampling_params"]["prompt_logprobs"]
                ids = [[pos * 10 + k for k in range(topk)] for pos in range(seq_len)]
                lps = [[-(pos + 0.1 * k) for k in range(topk)] for pos in range(seq_len)]
                return SimpleNamespace(extra_fields={"prompt_ids": ids, "prompt_logprobs": lps})

        trainer.get_llm_client = lambda: _FakeLLMClient()

        written: dict = {}

        def _fake_kv_batch_put(keys, partition_id, fields):
            written["keys"] = keys
            written["partition_id"] = partition_id
            written["fields"] = fields
            return KVBatchMeta(
                keys=keys,
                tags=[{}, {}, {}],
                partition_id=partition_id,
                fields=list(tq_data.keys()) + list(fields.keys()),
            )

        batch = KVBatchMeta(
            keys=["k0", "k1", "k2"],
            tags=[{}, {}, {}],
            partition_id="train",
            fields=list(tq_data.keys()),
        )
        metrics: dict[str, float] = {}

        with (
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_get", return_value=tq_data),
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_put", side_effect=_fake_kv_batch_put),
        ):
            out_batch = trainer._compute_self_teacher_log_probs(batch, metrics)

        # Only sample 0 qualified (winner sample 1 excluded itself, group g2 all failed).
        assert len(generate_calls) == 1
        call = generate_calls[0]
        assert call["image_data"] == ["pil_img_0"]
        assert call["audio_data"] == ["wav_aud_0"]
        assert call["mm_processor_kwargs"] == {"sampling_rate": 16000}
        assert call["sampling_params"] == {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 2,
            "detokenize": False,
        }

        fields = written["fields"]
        assert fields["self_distill_mask"].tolist() == [True, False, False]
        t_ids = list(fields["teacher_ids"].unbind())
        t_lps = list(fields["teacher_logprobs"].unbind())
        assert t_ids[0].shape == (len(p0) + len(r0), 2)
        assert t_ids[1].shape == (len(p1) + len(r1), 2)
        assert t_ids[2].shape == (len(p2) + len(r2), 2)
        assert (t_lps[1] == 0.0).all()
        assert (t_lps[2] == 0.0).all()
        assert (t_lps[0][len(p0) - 1 : len(p0) - 1 + len(r0)] != 0.0).all()

        assert metrics["self_distillation/valid_ratio"] == pytest.approx(1.0 / 3.0)
        assert metrics["self_distillation/fallback_ratio"] == pytest.approx(2.0 / 3.0)
        assert metrics["self_distillation/source_sibling_ratio"] == pytest.approx(1.0 / 3.0)
        assert metrics["self_distillation/mean_reprompt_delta_tokens"] > 0.0
        assert "self_distill_mask" in out_batch.fields

    def test_ema_teacher_source_swaps_syncs_and_restores_in_omni_sync(self):
        tok = _DummyTokenizer()
        trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
        trainer.tokenizer = tok
        trainer.processor = None
        trainer.global_steps = 3
        trainer.config = OmegaConf.create({"data": {"mm_processor_kwargs": {}}})
        trainer.distillation_config = OmniDistillationConfig(
            enabled=True,
            nnodes=0,
            n_gpus_per_node=0,
            distillation_loss=OmniDistillationLossConfig(
                loss_mode="forward_kl_topk",
                topk=2,
                use_policy_gradient=False,
            ),
            self_teacher=SelfDistillationConfig(
                enabled=True,
                mode="opsd",
                teacher_source="ema",
                ema_decay=0.9,
            ),
        )

        p0 = torch.tensor(tok.encode("<user> q1 <assistant>"), dtype=torch.long)
        r0 = torch.tensor(tok.encode("ans"), dtype=torch.long)
        tq_data = TensorDict(
            {
                "prompts": torch.nested.as_nested_tensor([p0], layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor([r0], layout=torch.jagged),
                "response_mask": torch.nested.as_nested_tensor([torch.ones_like(r0)], layout=torch.jagged),
                "rm_scores": torch.nested.as_nested_tensor([torch.tensor([0.0])], layout=torch.jagged),
                "uid": NonTensorStack("u1"),
                "raw_prompt": NonTensorStack([{"role": "user", "content": "q1"}]),
                "reward_model": NonTensorStack({"ground_truth": "42"}),
            },
            batch_size=[1],
        )

        events: list[str] = []

        class _FakeActorWG:
            def swap_ema_weights(self, use_ema: bool) -> bool:
                events.append(f"swap:{use_ema}")
                return True

        class _FakeCheckpointMgr:
            def update_weights(self, step: int) -> None:
                events.append(f"sync:{step}")

            def resume_generation_replicas(self) -> None:
                events.append("resume")

        class _FakeLLMClient:
            async def generate(self, **kwargs):
                events.append("generate")
                seq_len = len(kwargs["prompt_ids"])
                topk = kwargs["sampling_params"]["prompt_logprobs"]
                ids = [[pos * 10 + k for k in range(topk)] for pos in range(seq_len)]
                lps = [[-0.5 for _ in range(topk)] for _ in range(seq_len)]
                return SimpleNamespace(extra_fields={"prompt_ids": ids, "prompt_logprobs": lps})

        trainer.actor_rollout_wg = _FakeActorWG()
        trainer.checkpoint_manager = _FakeCheckpointMgr()
        trainer.get_llm_client = lambda: _FakeLLMClient()

        batch = KVBatchMeta(keys=["k0"], tags=[{}], partition_id="train", fields=list(tq_data.keys()))
        with (
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_get", return_value=tq_data),
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_put", return_value=batch),
        ):
            trainer._compute_self_teacher_log_probs(batch, {})

        assert events == ["swap:True", "sync:3", "resume", "generate", "swap:False"]

        # Verify live weights are still restored even when scoring raises.
        events.clear()

        class _FailingLLMClient:
            async def generate(self, **kwargs):
                events.append("generate_fail")
                raise RuntimeError("server error")

        trainer.get_llm_client = lambda: _FailingLLMClient()
        with (
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_get", return_value=tq_data),
            pytest.raises(RuntimeError, match="server error"),
        ):
            trainer._compute_self_teacher_log_probs(batch, {})
        assert events == ["swap:True", "sync:3", "resume", "generate_fail", "swap:False"]

    def test_separate_async_supports_policy_and_rejects_ema_teacher_source(self):
        tok = _DummyTokenizer()
        trainer = OmniPPOTrainerSeparateAsync.__new__(OmniPPOTrainerSeparateAsync)
        trainer.tokenizer = tok
        trainer.processor = None
        trainer.standalone_server_manager = object()
        trainer.config = OmegaConf.create({"data": {"mm_processor_kwargs": {}}})

        # EMA must raise clearly on separate_async to avoid overwriting concurrent rollout replicas.
        trainer.distillation_config = OmniDistillationConfig(
            enabled=True,
            nnodes=0,
            n_gpus_per_node=0,
            self_teacher=SelfDistillationConfig(enabled=True, mode="sdpo", teacher_source="ema"),
        )
        batch = KVBatchMeta(keys=["k0"], tags=[{}], partition_id="train")
        with pytest.raises(NotImplementedError, match="omni_separate_async"):
            trainer._compute_self_teacher_log_probs(batch, {})

        # Policy self-distillation works on separate_async via standalone get_llm_client().
        trainer.distillation_config = OmniDistillationConfig(
            enabled=True,
            nnodes=0,
            n_gpus_per_node=0,
            distillation_loss=OmniDistillationLossConfig(
                loss_mode="forward_kl_topk",
                topk=2,
                use_policy_gradient=False,
            ),
            self_teacher=SelfDistillationConfig(enabled=True, mode="opsd", teacher_source="policy"),
        )
        p0 = torch.tensor(tok.encode("<user> q1 <assistant>"), dtype=torch.long)
        r0 = torch.tensor(tok.encode("ans"), dtype=torch.long)
        tq_data = TensorDict(
            {
                "prompts": torch.nested.as_nested_tensor([p0], layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor([r0], layout=torch.jagged),
                "response_mask": torch.nested.as_nested_tensor([torch.ones_like(r0)], layout=torch.jagged),
                "rm_scores": torch.nested.as_nested_tensor([torch.tensor([0.0])], layout=torch.jagged),
                "uid": NonTensorStack("u1"),
                "raw_prompt": NonTensorStack([{"role": "user", "content": "q1"}]),
                "reward_model": NonTensorStack({"ground_truth": "42"}),
            },
            batch_size=[1],
        )

        class _FakeStandaloneClient:
            async def generate(self, **kwargs):
                seq_len = len(kwargs["prompt_ids"])
                topk = kwargs["sampling_params"]["prompt_logprobs"]
                ids = [[pos * 10 + k for k in range(topk)] for pos in range(seq_len)]
                lps = [[-0.25 for _ in range(topk)] for _ in range(seq_len)]
                return SimpleNamespace(extra_fields={"prompt_ids": ids, "prompt_logprobs": lps})

        trainer.get_llm_client = lambda: _FakeStandaloneClient()
        metrics: dict[str, float] = {}
        with (
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_get", return_value=tq_data),
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_put", return_value=batch),
        ):
            trainer._compute_self_teacher_log_probs(batch, metrics)
        assert metrics["self_distillation/valid_ratio"] == pytest.approx(1.0)
        assert metrics["self_distillation/source_reference_ratio"] == pytest.approx(1.0)

    def test_actor_worker_swap_ema_weights_and_update(self):
        model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.role = "actor_rollout"
        worker._ema_teacher_state = EMATeacherState(decay=0.8)
        worker._ema_teacher_state.initialize(model)
        worker._ema_live_backups = None
        worker.actor = SimpleNamespace(engine=SimpleNamespace(module=model))

        # Live weights advance to 2.0, EMA moves to 0.8*1.0 + 0.2*2.0 = 1.2.
        with torch.no_grad():
            model.weight.fill_(2.0)
        worker._ema_teacher_state.update(model)

        # Swap EMA in (idempotent on double-call), then restore live weights.
        assert worker.swap_ema_weights(use_ema=True) is True
        torch.testing.assert_close(model.weight, torch.full((2, 3), 1.2))
        assert worker.swap_ema_weights(use_ema=True) is True
        torch.testing.assert_close(model.weight, torch.full((2, 3), 1.2))

        assert worker.swap_ema_weights(use_ema=False) is True
        torch.testing.assert_close(model.weight, torch.full((2, 3), 2.0))
        assert worker.swap_ema_weights(use_ema=False) is False

    def test_update_actor_disables_distillation_only_when_fallback_to_pg_enabled(self):
        def _run_update(fallback_to_pg: bool) -> dict:
            trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
            trainer.config = OmegaConf.create(
                {
                    "actor_rollout_ref": {
                        "actor": {
                            "calculate_entropy": False,
                            "entropy_coeff": 0.0,
                            "ppo_mini_batch_size": 2,
                            "ppo_epochs": 1,
                            "data_loader_seed": 0,
                            "shuffle": False,
                        },
                        "rollout": {"n": 2, "temperature": 1.0},
                    },
                    "distillation": {"enabled": True, "self_teacher": {"enabled": True}},
                }
            )
            trainer.distillation_config = OmniDistillationConfig(
                enabled=True,
                nnodes=0,
                n_gpus_per_node=0,
                distillation_loss=OmniDistillationLossConfig(
                    loss_mode="forward_kl_topk",
                    topk=4,
                    use_task_rewards=False,
                    use_policy_gradient=False,
                ),
                self_teacher=SelfDistillationConfig(enabled=True, mode="sdpo", fallback_to_pg=fallback_to_pg),
            )
            trainer.actor_rollout_wg = MagicMock()
            trainer.actor_rollout_wg.update_actor.return_value = {"metrics": {"mfu": 0.1}}
            batch = KVBatchMeta(keys=["k0"], tags=[{}], partition_id="train")
            trainer._update_actor(batch, {})
            return batch.extra_info

        assert _run_update(fallback_to_pg=True)["distillation_only"] is False
        assert _run_update(fallback_to_pg=False)["distillation_only"] is True

    def test_self_distillation_task_runner_hides_distillation_from_agent_loop(self):
        seen_enabled: list[bool] = []

        class _FakeBaseTaskRunner:
            def __init__(self, config):
                self.config = config

            def init_agent_loop_manager(self):
                seen_enabled.append(bool(self.config.distillation.enabled))

        remote_cls = _build_self_distillation_task_runner_v1(_FakeBaseTaskRunner)
        runner_cls = remote_cls.__ray_metadata__.modified_class
        cfg = OmegaConf.create({"distillation": {"enabled": True, "self_teacher": {"enabled": True}}})
        runner = runner_cls(cfg)
        runner.init_agent_loop_manager()

        assert seen_enabled == [False]
        assert runner.config.distillation.enabled is True

    def test_summarize_metrics_empty_batch(self):
        assert summarize_self_distillation_metrics([]) == {}


class TestCodeRabbitReviewFixes:
    def test_empty_branch_loss_stays_finite_with_masked_neg_inf(self):
        from verl_omni.trainer.distillation.losses import _differentiable_zero

        log_probs = torch.tensor([[-1.0, float("-inf")]], requires_grad=True)
        zero = _differentiable_zero({"log_probs": log_probs}, fallback_device=torch.device("cpu"))
        assert torch.isfinite(zero)
        assert zero.item() == 0.0
        assert zero.requires_grad is True
        zero.backward()
        assert torch.isfinite(log_probs.grad).all()
        assert torch.equal(log_probs.grad, torch.zeros_like(log_probs))

    def test_omni_distillation_ppo_loss_forwards_megatron_packing_kwargs(self):
        captured: dict[str, object] = {}

        def _fake_distill(*args, **kwargs):
            captured["kwargs"] = kwargs
            return torch.tensor(0.0), {}

        distill_cfg = OmniDistillationConfig(
            enabled=True,
            nnodes=0,
            n_gpus_per_node=0,
            distillation_loss=OmniDistillationLossConfig(loss_mode="forward_kl_topk", topk=4),
            self_teacher=SelfDistillationConfig(enabled=True, mode="sdpo"),
        )
        with patch("verl_omni.trainer.distillation.losses.distillation_ppo_loss", side_effect=_fake_distill):
            omni_distillation_ppo_loss(
                config=SimpleNamespace(),
                distillation_config=distill_cfg,
                student_logits=torch.zeros(2, 4),
                data={},
                quant_align_size=16,
                cp_layout="thd",
                local_cp_size=2,
            )
        assert captured["kwargs"] == {"quant_align_size": 16, "cp_layout": "thd", "local_cp_size": 2}

    def test_preserve_multimodal_inputs_false_strips_media_blocks(self):
        tok = _DummyTokenizer()
        raw = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "file:///tmp/a.wav"},
                    {"type": "image", "image": "img_obj"},
                    {"type": "text", "text": "transcribe"},
                ],
            }
        ]
        res = build_multimodal_teacher_prompt(
            raw_prompt=raw,
            student_prompt_ids=tok.encode("<user> <audio> <image> transcribe <assistant>"),
            privileged_text="Correct answer: hello",
            tokenizer=tok,
            multi_modal_data={"audios": ["wav"], "images": ["img"]},
            config=SelfDistillationConfig(enabled=True, preserve_multimodal_inputs=False),
        )
        assert res.multi_modal_data == {}
        user_blocks = res.teacher_messages[0]["content"]
        assert all(b.get("type") == "text" for b in user_blocks)
        assert res.context_retained is True

    def test_trimming_removes_all_context_sets_context_retained_false(self):
        tok = _DummyTokenizer()
        student_ids = tok.encode("<user> question here <assistant>")
        res = build_multimodal_teacher_prompt(
            raw_prompt=[{"role": "user", "content": "question here"}],
            student_prompt_ids=student_ids,
            privileged_text="very long privileged context that cannot fit",
            tokenizer=tok,
            config=SelfDistillationConfig(enabled=True, max_reprompt_tokens=len(student_ids)),
        )
        assert res.context_retained is False
        assert res.trimmed_context is True

    def test_extract_inline_multimodal_preserves_audio_url(self):
        from verl_omni.trainer.omni.trainer_base import _extract_inline_multimodal_data

        raw = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "https://example.com/a.wav"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
                ],
            }
        ]
        extracted = _extract_inline_multimodal_data(raw)
        assert extracted == {
            "audios": ["https://example.com/a.wav"],
            "images": ["https://example.com/b.png"],
        }

    def test_extra_fields_reward_feedback_and_response_budget_reservation(self):
        tok = _DummyTokenizer()
        trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
        trainer.tokenizer = tok
        trainer.processor = None
        # Effective rollout limit = 40 tokens: sample 0 has response_len=2 (room for reprompt),
        # sample 1 has response_len=39 (no room -> must route to fallback without calling server).
        trainer.config = OmegaConf.create(
            {
                "data": {"mm_processor_kwargs": {}, "max_prompt_length": 20, "max_response_length": 20},
                "actor_rollout_ref": {"rollout": {"max_model_len": 40}},
            }
        )
        trainer.distillation_config = OmniDistillationConfig(
            enabled=True,
            nnodes=0,
            n_gpus_per_node=0,
            distillation_loss=OmniDistillationLossConfig(loss_mode="forward_kl_topk", topk=2),
            self_teacher=SelfDistillationConfig(enabled=True, mode="sdpo", fallback_to_pg=True),
        )

        p0 = torch.tensor(tok.encode("<user> q0 <assistant>"), dtype=torch.long)
        r0 = torch.tensor(tok.encode("a b"), dtype=torch.long)
        p1 = torch.tensor(tok.encode("<user> q1 <assistant>"), dtype=torch.long)
        r1 = torch.tensor([1] * 39, dtype=torch.long)
        tq_data = TensorDict(
            {
                "prompts": torch.nested.as_nested_tensor([p0, p1], layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor([r0, r1], layout=torch.jagged),
                "response_mask": torch.nested.as_nested_tensor(
                    [torch.ones_like(r0), torch.ones_like(r1)], layout=torch.jagged
                ),
                "rm_scores": torch.nested.as_nested_tensor(
                    [torch.tensor([0.0]), torch.tensor([0.0])], layout=torch.jagged
                ),
                "uid": NonTensorStack("u0", "u1"),
                "raw_prompt": NonTensorStack(
                    [{"role": "user", "content": "q0"}],
                    [{"role": "user", "content": "q1"}],
                ),
                "extra_fields": NonTensorStack(
                    {"reward_extra_info": {"feedback": "fix"}},
                    {"reward_extra_info": {"feedback": "fix"}},
                ),
            },
            batch_size=[2],
        )

        seen_prompts: list[int] = []

        class _FakeLLMClient:
            async def generate(self, **kwargs):
                seen_prompts.append(len(kwargs["prompt_ids"]))
                seq_len = len(kwargs["prompt_ids"])
                ids = [[0, 1] for _ in range(seq_len)]
                lps = [[-0.5, -0.5] for _ in range(seq_len)]
                return SimpleNamespace(extra_fields={"prompt_ids": ids, "prompt_logprobs": lps})

        trainer.get_llm_client = lambda: _FakeLLMClient()
        captured: dict[str, object] = {}

        def _fake_put(keys, partition_id, fields):
            captured["fields"] = fields
            return KVBatchMeta(keys=keys, tags=[{}, {}], partition_id=partition_id, fields=list(fields.keys()))

        batch = KVBatchMeta(keys=["k0", "k1"], tags=[{}, {}], partition_id="train", fields=list(tq_data.keys()))
        metrics: dict[str, float] = {}
        with (
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_get", return_value=tq_data),
            patch("verl_omni.trainer.omni.trainer_base.tq.kv_batch_put", side_effect=_fake_put),
        ):
            trainer._compute_self_teacher_log_probs(batch, metrics)

        # Sample 0 received feedback from extra_fields and was scored; sample 1 routed to fallback.
        assert len(seen_prompts) == 1
        assert seen_prompts[0] < 40
        assert captured["fields"]["self_distill_mask"].tolist() == [True, False]
        assert metrics["self_distillation/valid_ratio"] == pytest.approx(0.5)
        assert metrics["self_distillation/fallback_ratio"] == pytest.approx(0.5)

    def test_ema_swap_rolls_back_on_partial_copy_failure_and_checkpoint_round_trip(self, tmp_path):
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[1].weight.fill_(2.0)

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.role = "actor"
        worker.actor = SimpleNamespace(
            engine=SimpleNamespace(module=model),
            save_checkpoint=lambda *a, **kw: None,
            load_checkpoint=lambda *a, **kw: None,
        )
        worker._ema_teacher_state = EMATeacherState(decay=0.5)
        worker._ema_teacher_state.initialize(model)
        worker._ema_live_backups = None

        with torch.no_grad():
            model[0].weight.fill_(3.0)
            model[1].weight.fill_(4.0)
        worker._ema_teacher_state.update(model)

        # Save EMA state alongside checkpoint, mutate EMA, then restore via load_checkpoint.
        ckpt_dir = str(tmp_path / "ckpt_1")
        worker.save_checkpoint(ckpt_dir)
        saved_shadow_0 = worker._ema_teacher_state.shadow["0.weight"].clone()
        worker._ema_teacher_state.shadow["0.weight"].fill_(999.0)
        worker.load_checkpoint(ckpt_dir)
        assert torch.allclose(worker._ema_teacher_state.shadow["0.weight"], saved_shadow_0)

        # Inject a shape mismatch on the second param to force a copy failure mid-swap.
        worker._ema_teacher_state.shadow["1.weight"] = torch.zeros(3, 3)
        with pytest.raises(RuntimeError):
            worker.swap_ema_weights(use_ema=True)
        # First parameter must be rolled back to 3.0 and backup must stay None.
        assert torch.allclose(model[0].weight, torch.full_like(model[0].weight, 3.0))
        assert worker._ema_live_backups is None

    def test_update_actor_tolerates_missing_ema_attribute(self):
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.config = OmegaConf.create({"actor": {}})
        worker._trajectory_materializer = None
        worker.enable_routing_replay = False
        worker.actor = SimpleNamespace(
            train_mini_batch=lambda data: TensorDict({"metrics": torch.tensor([1.0])}, batch_size=[])
        )
        out = worker.update_actor(TensorDict({}, batch_size=[]))
        assert out is not None
