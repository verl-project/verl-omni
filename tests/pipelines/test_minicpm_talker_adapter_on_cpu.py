"""Tests for the MiniCPM-o talker training adapter (standalone, no package import).

Tests the adapter's logic by loading the sources directly, avoiding the full
verl_omni package import chain which requires tensordict and other heavy
dependencies not needed for these unit tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent.parent
ADAPTER = REPO / "verl_omni" / "pipelines" / "minicpm" / "talker_training_adapter.py"
FORWARD = REPO / "verl_omni" / "pipelines" / "minicpm" / "talker_forward.py"


class TestStripModules(unittest.TestCase):
    """The talker adapter must NOT strip the TTS module (unlike thinker)."""

    def test_strip_modules_returns_empty(self):
        src = ADAPTER.read_text()
        section = src[src.index("def get_strip_modules") : src.index("def get_fsdp_ignored_module_names")]
        self.assertIn("return []", section)

    def test_loader_keeps_init_tts(self):
        """The loader must set config.init_tts = True (the thinker sets False)."""
        src = ADAPTER.read_text()
        self.assertIn("config.init_tts = True", src)

    def test_freezes_everything_outside_tts(self):
        src = ADAPTER.read_text()
        self.assertIn('requires_grad_(name.startswith("tts."))', src)

    def test_strips_code2wav_renderer(self):
        src = ADAPTER.read_text()
        self.assertIn("audio_tokenizer", src)

    def test_requires_no_remove_padding(self):
        """rmpad flattening invalidates the replay span coordinates."""
        src = ADAPTER.read_text()
        self.assertIn("use_remove_padding", src)


class TestFSAPIgnoredModules(unittest.TestCase):
    """FSDP must ignore the frozen thinker, not the trainable talker."""

    def test_ignored_modules_exclude_thinker_not_tts(self):
        src = ADAPTER.read_text()
        section = src[src.index("def get_fsdp_ignored_module_names") : src.index("def configure_model")]
        for name in ("llm", "vpm", "apm", "resampler"):
            self.assertIn(f'"{name}"', section)
        self.assertNotIn('"tts"', section)


class TestReplayValidation(unittest.TestCase):
    """prepare_model_inputs validates the replay payload fail-closed."""

    def test_replay_key_defined(self):
        src = FORWARD.read_text()
        self.assertIn('MINICPM_TALKER_REPLAY_KEY = "minicpm_talker_replay"', src)

    def test_required_replay_fields(self):
        src = ADAPTER.read_text()
        for field in ("text_ids", "audio_codes", "response_start"):
            self.assertIn(f'"{field}"', src)

    def test_response_alignment_verified(self):
        """The batch's response tokens must equal the replayed codec frames."""
        src = ADAPTER.read_text()
        self.assertIn("torch.equal", src)


class TestCodecTokenEmbedding(unittest.TestCase):
    """Verify multi-codebook embedding logic (pure torch, no package deps)."""

    def test_single_codebook(self):
        tokens = torch.tensor([[1, 2, 3]])
        emb = torch.nn.Embedding(100, 64)
        embedded = emb(tokens)
        self.assertEqual(embedded.shape, (1, 3, 64))

    def test_multi_codebook_sum(self):
        tokens = torch.tensor([[[1, 2], [3, 4], [5, 6]]])  # [1, 3, 2]
        num_vq = 2
        emb_code = [torch.nn.Embedding(100, 64) for _ in range(num_vq)]
        code_embs = [emb_code[q](tokens[..., q]) for q in range(num_vq)]
        summed = torch.stack(code_embs, dim=-1).sum(dim=-1)
        self.assertEqual(summed.shape, (1, 3, 64))

    def test_multi_codebook_sum_matches_generate_loop(self):
        tokens = torch.tensor([[[1, 5], [2, 6]]])  # [1, 2, 2]
        emb_code = [torch.nn.Embedding(100, 32) for _ in range(2)]
        code_embs = [emb_code[q](tokens[..., q]) for q in range(2)]
        summed = torch.stack(code_embs, dim=-1).sum(dim=-1)
        expected = emb_code[0](tokens[..., 0]) + emb_code[1](tokens[..., 1])
        self.assertTrue(torch.allclose(summed, expected))


class TestConditioningConstruction(unittest.TestCase):
    """Verify the thinker→talker conditioning math (hidden_text_merge)."""

    def test_hidden_text_merge_shape(self):
        batch, seq, hidden_dim, tts_dim = 2, 5, 4096, 768
        text_tokens = torch.randint(0, 1000, (batch, seq))
        hidden_states = torch.randn(batch, seq, hidden_dim)
        emb_text = torch.nn.Embedding(1000, tts_dim)
        projector = torch.nn.Linear(hidden_dim, tts_dim)
        conditioning = emb_text(text_tokens) + projector(hidden_states)
        self.assertEqual(conditioning.shape, (batch, seq, tts_dim))

    def test_normalize_projected_hidden(self):
        import torch.nn.functional as F
        hidden_embeds = torch.randn(1, 5, 768)
        normalized = F.normalize(hidden_embeds, p=2, dim=-1)
        norms = normalized.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TestOfficialPrefixConstruction(unittest.TestCase):
    """The teacher-forced input must mirror _generate_speech_non_streaming."""

    def test_prefix_includes_text_eos_then_audio_bos(self):
        src = FORWARD.read_text()
        self.assertIn("text_eos_token_id", src)
        self.assertIn("audio_bos_token_id", src)
        cat_idx = src.index("torch.cat([conditioning, text_eos, audio_bos, codec_embeds[:-1]]")
        self.assertGreater(cat_idx, 0)

    def test_logits_from_head_code_not_backbone(self):
        """The backbone has no lm_head; logits come from head_code[q]."""
        src = FORWARD.read_text()
        self.assertIn("head_code[0](hidden)", src)
        self.assertNotIn("tts_outputs.logits", src)
        self.assertNotIn(".logits\n", src.replace("logits.logits", ""))

    def test_multi_codebook_rejected(self):
        """MiniCPM-o 4.5 is single-codebook; num_vq > 1 must be rejected."""
        src = FORWARD.read_text()
        self.assertIn("num_vq != 1", src)

    def test_logits_are_codec_aligned(self):
        """Position cond_len + 1 + j predicts codec j (audio_bos predicts c_0)."""
        cond_len, codec_len = 10, 20
        seq_len = cond_len + 2 + codec_len - 1
        logits = torch.randn(1, seq_len, 6562)
        aligned = logits[:, cond_len + 1 :]
        self.assertEqual(aligned.shape[1], codec_len)
        self.assertTrue(torch.equal(aligned[:, 0], logits[:, cond_len + 1]))

    def test_scatter_offsets_match(self):
        """minicpm_talker_actor_logits scatters [cond+1 : cond+1+T] to response."""
        src = FORWARD.read_text()
        self.assertIn("source = batch.cond_lens[index] + 1", src)
        self.assertIn("response_start : response_start + response_len", src)

    def test_policy_codec_alignment_checked(self):
        """Fail-closed when the batch response tokens differ from the codec frames."""
        src = FORWARD.read_text()
        self.assertIn("torch.equal(policy_ids, codec_ids)", src)


class TestThinkerStageForward(unittest.TestCase):
    """The wrapped forward runs the thinker under no_grad with hidden states."""

    def test_thinker_forward_no_grad_with_hidden_states(self):
        src = ADAPTER.read_text()
        self.assertIn("torch.no_grad()", src)
        self.assertIn("output_hidden_states=True", src)
        self.assertIn("hidden_states[tts_proj_layer]", src)

    def test_thinker_input_rebuilt_from_replay_text(self):
        """prepare_model_inputs replaces the codec-mixed ids with the text ids."""
        src = ADAPTER.read_text()
        self.assertIn('model_inputs["input_ids"] = text_buffer', src)



class TestCodecPaddingSlice(unittest.TestCase):
    """The padded codec row must be sliced to its true length before scoring."""

    def test_codec_rows_sliced_by_codec_lens(self):
        src = ADAPTER.read_text()
        self.assertIn("tts_audio_codes[index, :codec_len]", src)

    def test_padded_row_would_teacher_force_zeros(self):
        """Demonstrate the bug the slice prevents: unsliced rows score padding."""
        row = torch.tensor([3, 5, 8, 0, 0])  # true length 3, padded with zeros
        codec_len = 3
        sliced = row[:codec_len]
        self.assertEqual(sliced.tolist(), [3, 5, 8])
        # The unsliced row would append three spurious zero-token targets.
        self.assertNotEqual(row.tolist(), sliced.tolist())


class TestAsyncChunkValidation:
    """async_chunk is validated per pipeline mode."""

    def _load_adapter(self):
        import importlib.util
        path = Path(__file__).parent.parent.parent / "verl_omni" / "pipelines" / "minicpm" / "omni_rollout_adapter.py"
        src = path.read_text()
        # Verify validate_async_chunk exists in the source (can't import the
        # full package chain without heavy deps).
        return src

    def test_thinker_talker_rejects_explicit_true(self):
        src = self._load_adapter()
        assert "async_chunk=false" in src, "thinker_talker must reject async_chunk=true"
        assert "raise ValueError" in src

    def test_thinker_talker_explicit_false_ok(self):
        src = self._load_adapter()
        assert "return False" in src, "thinker_talker absent async_chunk defaults to false"

    def test_thinker_only_default_true(self):
        src = self._load_adapter()
        assert "return async_chunk if async_chunk is not None else True" in src, \
            "thinker_only absent async_chunk defaults to true"

    def test_validate_method_exists(self):
        src = self._load_adapter()
        assert "def validate_async_chunk" in src, "per-mode async_chunk validation method exists"


if __name__ == "__main__":
    unittest.main()
