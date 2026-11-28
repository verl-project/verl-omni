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
"""CPU behavior tests for the MiniCPM-o talker training pipeline.

The teacher-forced math and the adapter/rollout contracts import the full
verl_omni stack (the forward lives in talker_training_adapter since the
merge) and are skipped when it is not installed.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).parent.parent.parent


_STACK_MISSING = importlib.util.find_spec("verl") is None or importlib.util.find_spec("vllm_omni") is None


def _load_stack_module(name: str):
    """Import a verl_omni module, skipping the test when the stack is absent."""
    if _STACK_MISSING:
        raise unittest.SkipTest("requires the verl_omni dependency stack")
    return importlib.import_module(name)


class _StubTTS(torch.nn.Module):
    """A minimal ``model.tts`` with the modules the forward math touches."""

    def __init__(
        self,
        hidden_size: int = 32,
        llm_hidden: int = 16,
        num_audio_tokens: int = 100,
        normalize_projected_hidden: bool = True,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.emb_text = torch.nn.Embedding(50, hidden_size)
        self.projector_semantic = torch.nn.Linear(llm_hidden, hidden_size)
        self.emb_code = [torch.nn.Embedding(num_audio_tokens, hidden_size)]
        self.head_code = [torch.nn.Linear(hidden_size, num_audio_tokens)]
        self.config = SimpleNamespace(normalize_projected_hidden=normalize_projected_hidden)
        self.model = self

    def forward(self, **kwargs):
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])


def _stub_model(normalize_projected_hidden: bool = True):
    tts = _StubTTS(normalize_projected_hidden=normalize_projected_hidden)
    return SimpleNamespace(
        tts=tts,
        config=SimpleNamespace(
            tts_config=SimpleNamespace(
                text_eos_token_id=11,
                audio_bos_token_id=12,
                num_vq=1,
                num_audio_tokens=100,
                hidden_size=32,
                normalize_projected_hidden=normalize_projected_hidden,
            )
        ),
    )


class _ForwardCase(unittest.TestCase):
    """Base for the teacher-forced math tests: loads the merged adapter."""

    forward = None

    @classmethod
    def setUpClass(cls):
        cls.forward = _load_stack_module("verl_omni.pipelines.minicpm.talker_training_adapter")

    @property
    def mod(self):
        return self.__class__.forward


class TestTalkerTokens(_ForwardCase):
    """TalkerTokens.from_config extracts the scalar constants."""

    def test_from_config(self):
        tokens = self.mod.TalkerTokens.from_config(_stub_model().config)
        self.assertEqual(tokens.text_eos_token_id, 11)
        self.assertEqual(tokens.audio_bos_token_id, 12)
        self.assertEqual(tokens.num_vq, 1)
        self.assertEqual(tokens.num_audio_tokens, 100)
        self.assertEqual(tokens.hidden_size, 32)
        self.assertTrue(tokens.normalize_projected_hidden)


class TestConditioning(_ForwardCase):
    """The hidden_text_merge conditioning follows the official math."""

    def test_sum_without_normalization(self):
        model = _stub_model(normalize_projected_hidden=False)
        text = torch.randint(0, 50, (1, 4))
        hidden = torch.randn(1, 4, 16)
        got = self.mod.build_conditioning(model, text, hidden)
        expected = model.tts.emb_text(text) + model.tts.projector_semantic(hidden)
        self.assertTrue(torch.allclose(got, expected))

    def test_normalized_when_configured(self):
        model = _stub_model(normalize_projected_hidden=True)
        text = torch.randint(0, 50, (1, 4))
        hidden = torch.randn(1, 4, 16)
        got = self.mod.build_conditioning(model, text, hidden)
        projected = torch.nn.functional.normalize(model.tts.projector_semantic(hidden), p=2, dim=-1)
        self.assertTrue(torch.allclose(got, model.tts.emb_text(text) + projected))


class TestBuildTalkerBatch(_ForwardCase):
    """The teacher-forced prefix is [conditioning, text_eos, audio_bos, codec[:-1]]."""

    def _build(self, codec_len: int = 5, span: int = 4):
        model = _stub_model()
        tokens = self.mod.TalkerTokens.from_config(model.config)
        text = torch.randint(0, 50, (span,))
        hidden = torch.randn(span, 16)
        codes = torch.randint(0, 100, (codec_len,))
        batch = self.mod.build_talker_batch(model, [text], [hidden], [codes], tokens)
        return model, tokens, text, hidden, codes, batch

    def test_prefix_length(self):
        model, tokens, text, hidden, codes, batch = self._build(codec_len=5, span=4)
        self.assertEqual(batch.input_embeds.shape, (1, 4 + 2 + 4, tokens.hidden_size))
        self.assertEqual(batch.cond_lens, [4])
        self.assertEqual(batch.codec_lens, [5])

    def test_prefix_content(self):
        model, tokens, text, hidden, codes, batch = self._build(codec_len=5, span=4)
        conditioning = self.mod.build_conditioning(model, text.unsqueeze(0), hidden.unsqueeze(0))[0]
        text_eos = model.tts.emb_text(torch.tensor([tokens.text_eos_token_id]))
        audio_bos = model.tts.emb_text(torch.tensor([tokens.audio_bos_token_id]))
        codec_embeds = model.tts.emb_code[0](codes)
        expected = torch.cat([conditioning, text_eos, audio_bos, codec_embeds[:-1]], dim=0)
        self.assertTrue(torch.equal(batch.input_embeds[0], expected))
        self.assertTrue(torch.equal(batch.attention_mask, torch.ones(1, 10, dtype=torch.long)))

    def test_mismatched_batches_rejected(self):
        model = _stub_model()
        tokens = self.mod.TalkerTokens.from_config(model.config)
        with self.assertRaises(ValueError):
            self.mod.build_talker_batch(model, [torch.ones(3)], [], [torch.ones(2)], tokens)

    def test_multi_codebook_rejected(self):
        model = _stub_model()
        model.config.tts_config.num_vq = 2
        tokens = self.mod.TalkerTokens.from_config(model.config)
        with self.assertRaises(NotImplementedError):
            self.mod.build_talker_batch(model, [torch.ones(3)], [torch.randn(3, 16)], [torch.ones(2)], tokens)

    def test_out_of_vocab_codec_rejected(self):
        model = _stub_model()
        tokens = self.mod.TalkerTokens.from_config(model.config)
        with self.assertRaises(ValueError):
            self.mod.build_talker_batch(model, [torch.ones(3)], [torch.randn(3, 16)], [torch.tensor([999])], tokens)

    def test_empty_codec_rejected(self):
        model = _stub_model()
        tokens = self.mod.TalkerTokens.from_config(model.config)
        with self.assertRaises(ValueError):
            self.mod.build_talker_batch(
                model, [torch.ones(3)], [torch.randn(3, 16)], [torch.tensor([], dtype=torch.long)], tokens
            )

    def test_padding_across_rows(self):
        model = _stub_model()
        tokens = self.mod.TalkerTokens.from_config(model.config)
        texts = [torch.randint(0, 50, (4,)), torch.randint(0, 50, (2,))]
        hiddens = [torch.randn(4, 16), torch.randn(2, 16)]
        codes = [torch.randint(0, 100, (5,)), torch.randint(0, 100, (3,))]
        batch = self.mod.build_talker_batch(model, texts, hiddens, codes, tokens)
        self.assertEqual(batch.input_embeds.shape[0], 2)
        short_len = 2 + 2 + 2
        self.assertEqual(int(batch.attention_mask[1, :short_len].sum()), short_len)
        self.assertEqual(int(batch.attention_mask[1, short_len:].sum()), 0)


class TestTalkerLogits(_ForwardCase):
    """talker_logits routes the backbone output through head_code[0]."""

    def test_shape_and_source(self):
        model = _stub_model()
        tokens = self.mod.TalkerTokens.from_config(model.config)
        batch = self.mod.build_talker_batch(
            model,
            [torch.randint(0, 50, (4,))],
            [torch.randn(4, 16)],
            [torch.randint(0, 100, (5,))],
            tokens,
        )
        logits = self.mod.talker_logits(model, batch)
        self.assertEqual(logits.shape, (1, batch.input_embeds.shape[1], tokens.num_audio_tokens))
        # The stub backbone returns inputs_embeds, so head_code[0] applied to
        # the first row must match.
        self.assertTrue(torch.allclose(logits[0, 0], model.tts.head_code[0](batch.input_embeds[0, 0]), atol=1e-5))


class TestActorLogitsAlignment(_ForwardCase):
    """minicpm_talker_actor_logits scatters codec distributions onto responses."""

    def _run(self, codec_len: int = 5, span: int = 4):
        model = _stub_model()
        text = torch.randint(0, 50, (span,))
        hidden = torch.randn(span, 16)
        codes = torch.randint(0, 100, (codec_len,))
        input_ids = torch.zeros(1, 1 + codec_len, dtype=torch.long)
        input_ids[0, 1:] = codes
        attention_mask = torch.ones_like(input_ids)
        aligned = self.mod.minicpm_talker_actor_logits(model, input_ids, attention_mask, [text], [hidden], [codes])
        return model, codes, aligned

    def test_response_region_filled(self):
        model, codes, aligned = self._run(codec_len=5)
        self.assertEqual(aligned.shape, (1, 6, 100))
        self.assertFalse(torch.count_nonzero(aligned[0, 0]))
        self.assertTrue(torch.count_nonzero(aligned[0, 1:]))

    def test_alignment_off_by_one_is_shifted(self):
        # audio_bos at cond_len + 1 predicts codec_0, so response position j
        # must carry the logits of sequence position cond_len + 1 + j.
        model = _stub_model()
        span, codec_len = 4, 5
        text = torch.randint(0, 50, (span,))
        hidden = torch.randn(span, 16)
        codes = torch.randint(0, 100, (codec_len,))
        input_ids = torch.zeros(1, 1 + codec_len, dtype=torch.long)
        input_ids[0, 1:] = codes
        aligned = self.mod.minicpm_talker_actor_logits(
            model, input_ids, torch.ones_like(input_ids), [text], [hidden], [codes]
        )
        tokens = self.mod.TalkerTokens.from_config(model.config)
        cond_len = span  # conditioning length equals the text span
        logits = self.mod.talker_logits(
            model,
            self.mod.build_talker_batch(model, [text], [hidden], [codes], tokens),
        )
        self.assertTrue(torch.equal(aligned[0, 1], logits[0, cond_len + 1]))
        self.assertTrue(torch.equal(aligned[0, 5], logits[0, cond_len + 5]))

    def test_mismatched_codec_falls_back_right_aligned(self):
        # A batch whose response tokens disagree with the codec ids must not
        # crash; it falls back to the right-aligned span.
        model = _stub_model()
        text = torch.randint(0, 50, (4,))
        hidden = torch.randn(4, 16)
        codes = torch.randint(0, 100, (5,))
        input_ids = torch.zeros(1, 1 + 5, dtype=torch.long)  # all zeros != codes
        aligned = self.mod.minicpm_talker_actor_logits(
            model, input_ids, torch.ones_like(input_ids), [text], [hidden], [codes]
        )
        self.assertEqual(aligned.shape, (1, 6, 100))
        self.assertTrue(torch.count_nonzero(aligned[0, 1:]))


@unittest.skipUnless(
    importlib.util.find_spec("verl") is not None and importlib.util.find_spec("vllm_omni") is not None,
    "requires the verl_omni dependency stack",
)
class TestTalkerAdapterContract(unittest.TestCase):
    """Adapter-level contracts against the imported package modules."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = importlib.import_module("verl_omni.pipelines.minicpm.talker_training_adapter")
        cls.forward = importlib.import_module("verl_omni.pipelines.minicpm.talker_training_adapter")

    def test_replay_key_constant(self):
        self.assertEqual(self.forward.MINICPM_TALKER_REPLAY_KEY, "minicpm_talker_replay")

    def test_strip_modules_keeps_tts(self):
        self.assertEqual(self.adapter.MiniCPMTalkerAdapter.get_strip_modules(None), [])

    def test_fsdp_ignores_thinker_not_tts(self):
        ignored = self.adapter.MiniCPMTalkerAdapter.get_fsdp_ignored_module_names(None)
        self.assertEqual(set(ignored), {"llm", "vpm", "apm", "resampler"})

    def test_loader_requires_trust_remote_code(self):
        with self.assertRaises(ValueError):
            self.adapter.MiniCPMOWithTTS.from_pretrained("unused-path")

    def test_configure_model_rejects_remove_padding(self):
        config = SimpleNamespace(use_remove_padding=True)
        with self.assertRaises(ValueError):
            self.adapter.MiniCPMTalkerAdapter.configure_model(None, config)

    def test_prepare_model_inputs_requires_replay_payload(self):
        from tensordict import NonTensorStack, TensorDict

        model_inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4)}
        micro_batch = TensorDict({"extra_fields": NonTensorStack([{}])}, batch_size=[1])
        with self.assertRaises(RuntimeError):
            self.adapter.MiniCPMTalkerAdapter.prepare_model_inputs(model_inputs, micro_batch, None)


@unittest.skipUnless(
    importlib.util.find_spec("verl") is not None and importlib.util.find_spec("vllm_omni") is not None,
    "requires the verl_omni dependency stack",
)
class TestAsyncChunkValidation(unittest.TestCase):
    """async_chunk is validated per pipeline mode."""

    @classmethod
    def setUpClass(cls):
        cls.rollout = importlib.import_module("verl_omni.pipelines.minicpm.omni_rollout_adapter")

    def test_thinker_talker_rejects_explicit_true(self):
        with self.assertRaises(ValueError):
            self.rollout.MiniCPMORolloutAdapter.validate_async_chunk("thinker_talker", True)

    def test_thinker_talker_explicit_false_ok(self):
        self.assertFalse(self.rollout.MiniCPMORolloutAdapter.validate_async_chunk("thinker_talker", False))

    def test_thinker_talker_absent_defaults_false(self):
        self.assertFalse(self.rollout.MiniCPMORolloutAdapter.validate_async_chunk("thinker_talker", None))

    def test_thinker_only_absent_defaults_true(self):
        self.assertTrue(self.rollout.MiniCPMORolloutAdapter.validate_async_chunk("thinker_only", None))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.rollout.MiniCPMORolloutAdapter.validate_async_chunk("full", None)


if __name__ == "__main__":
    unittest.main()
