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
"""Qwen3-TTS talker patches: register the qwen-tts model classes for verl's FSDP actor, install
a teacher-forced codec-0 forward, freeze everything but the talker, and inject the per-sample
talker inputs through the agent loop.

Loaded via actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_tts. The
clone voice comes from actor_rollout_ref.model.override_config.tts_spk_embed_path.

The custom forward reads four extra kwargs injected per sample via multi_modal_inputs:
tts_text_ids (B, T_text), tts_audio_codes (B, R, 16), response_len (B,) and text_len (B,).
"""

import logging

import torch

from verl_omni.models.transformers.qwen3_tts_forward import (
    TEXT_PROMPT_TRAILER_TOKENS,
    build_assistant_text,
    load_speaker_xvector,
    tts_actor_logits,
)

logger = logging.getLogger(__name__)

# Only these submodule prefixes stay trainable; the rest of the model is frozen.
_TRAINABLE_PREFIXES = ("talker.model.", "talker.codec_head.")


def _speaker_embedding(model, batch_size: int, device, dtype) -> torch.Tensor:
    """Cached (B, D) speaker x-vector for the talker's speaker slot."""
    cache = getattr(model, "_verl_tts_spk_cache", None)
    if cache is None:
        path = getattr(model.config, "tts_spk_embed_path", None)
        if not path:
            raise RuntimeError(
                "Qwen3-TTS actor needs a precomputed clone x-vector: set "
                "actor_rollout_ref.model.override_config.tts_spk_embed_path to the JSON produced "
                "by the example's precompute_spk_embed.py."
            )
        cache = load_speaker_xvector(path)
        model._verl_tts_spk_cache = cache
        logger.info("verl_omni.qwen3_tts: loaded %d-dim speaker x-vector.", cache.shape[-1])
    return cache.to(device=device, dtype=dtype).expand(batch_size, -1)


def _reinit_rope_buffers(model) -> None:
    """Recompute rotary inv_freq buffers. verl materializes the model from the meta device via
    to_empty, which leaves non-persistent buffers as uninitialized memory; the weights load does
    not cover them, so every rotary embedding must be re-initialized after materialization."""
    for module in model.modules():
        rope_init = getattr(module, "rope_init_fn", None)
        inv_freq = getattr(module, "inv_freq", None)
        if rope_init is None or not torch.is_tensor(inv_freq):
            continue
        new_inv_freq, attention_scaling = rope_init(module.config, device=inv_freq.device)
        module.inv_freq.data.copy_(new_inv_freq.to(device=inv_freq.device, dtype=inv_freq.dtype))
        module.attention_scaling = attention_scaling


def _qwen3_tts_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    tts_text_ids=None,
    tts_audio_codes=None,
    response_len=None,
    text_len=None,
    **kwargs,
):
    """Return codec-0 logits aligned to verl's flat input_ids (see qwen3_tts_forward)."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if tts_audio_codes is None or response_len is None or text_len is None or tts_text_ids is None:
        raise RuntimeError(
            "qwen3_tts forward requires tts_text_ids/tts_audio_codes/response_len/text_len via "
            "multi_modal_inputs; missing keys mean the agent-loop patch did not run."
        )

    if not getattr(self, "_verl_rope_reinit_done", False):
        _reinit_rope_buffers(self)
        self._verl_rope_reinit_done = True

    B = input_ids.shape[0]
    spk = _speaker_embedding(self, B, input_ids.device, next(self.talker.parameters()).dtype)
    out_logits = tts_actor_logits(
        self, input_ids, attention_mask, tts_text_ids, tts_audio_codes, response_len, text_len, spk
    )
    return CausalLMOutputWithPast(logits=out_logits)


def _qwen3_tts_get_input_embeddings(self):
    return self.talker.model.codec_embedding


def _qwen3_tts_set_input_embeddings(self, value):
    self.talker.model.codec_embedding = value


def _mirror_talker_config(config) -> None:
    """Mirror standard transformer fields from talker_config to the top-level config, where
    verl's generic init path expects them."""
    tc = getattr(config, "talker_config", None)
    if tc is None:
        return
    for attr in ("num_attention_heads", "num_key_value_heads", "hidden_size", "num_hidden_layers"):
        if getattr(config, attr, None) is None and getattr(tc, attr, None) is not None:
            setattr(config, attr, getattr(tc, attr))


def _apply_freeze(model) -> None:
    """Train only the talker backbone and codec head; everything else is used but frozen."""
    n_train = 0
    for name, p in model.named_parameters():
        train = name.startswith(_TRAINABLE_PREFIXES)
        p.requires_grad_(train)
        n_train += int(train)
    logger.info("verl_omni.qwen3_tts: %d trainable param tensors (talker.model + codec_head).", n_train)


def _register_qwen3_tts_automodel() -> None:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        return

    # The trainable class lives in the qwen-tts package, not transformers.
    model_cls = config_cls = None
    for mod_path, attr in (
        ("transformers", "Qwen3TTSForConditionalGeneration"),
        ("qwen_tts.core.models.modeling_qwen3_tts", "Qwen3TTSForConditionalGeneration"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[attr])
        except Exception:  # noqa: BLE001
            continue
        model_cls = getattr(mod, attr, None)
        if model_cls is not None:
            break
    for mod_path in ("transformers", "qwen_tts.core.models.configuration_qwen3_tts"):
        try:
            mod = __import__(mod_path, fromlist=["Qwen3TTSConfig"])
        except Exception:  # noqa: BLE001
            continue
        config_cls = getattr(mod, "Qwen3TTSConfig", None)
        if config_cls is not None:
            break
    if model_cls is None:
        logger.warning("verl_omni.qwen3_tts: Qwen3TTSForConditionalGeneration not found; patch is a no-op.")
        return

    try:
        from verl.utils.model import _architecture_to_auto_class

        _architecture_to_auto_class.setdefault(model_cls.__name__, AutoModelForCausalLM)
    except ImportError as e:
        logger.warning("verl_omni.qwen3_tts: could not register architecture lookup (%s).", e)

    model_cls.forward = _qwen3_tts_forward
    model_cls.get_input_embeddings = _qwen3_tts_get_input_embeddings
    model_cls.set_input_embeddings = _qwen3_tts_set_input_embeddings
    model_cls._no_split_modules = ["Qwen3TTSTalkerDecoderLayer", "Qwen3TTSDecoderLayer"]

    _orig_post_init = getattr(model_cls, "post_init", None)

    def _post_init_with_freeze(self):
        if _orig_post_init is not None:
            _orig_post_init(self)
        _mirror_talker_config(self.config)
        _apply_freeze(self)

    model_cls.post_init = _post_init_with_freeze

    if config_cls is not None:
        # tie_word_embeddings=True disables FSDP meta-tensor init.
        class _FalseTie:
            def __get__(self, obj, objtype=None):
                return False

            def __set__(self, obj, value):
                pass

        config_cls.tie_word_embeddings = _FalseTie()
        # The HF repo ships no modeling code and importing qwen_tts does not self-register.
        try:
            from transformers import AutoConfig

            AutoConfig.register(getattr(config_cls, "model_type", "qwen3_tts"), config_cls)
        except ValueError:
            pass
        try:
            AutoModelForCausalLM.register(config_cls, model_cls)
        except ValueError:
            pass
        _make_config_save_non_fatal(config_cls)
    logger.info("verl_omni.qwen3_tts: installed talker codec-0 forward + freeze on %s.", model_cls.__name__)


def _make_config_save_non_fatal(config_cls) -> None:
    """Guard save_pretrained on the composite Qwen3-TTS config. verl's checkpoint manager writes
    the HF config unconditionally, and the nested talker/code2wav/speech_tokenizer config trips
    transformers' to_diff_dict with KeyError('dtype'), which would kill the run at the first save
    after the model and optim shards already wrote. Those shards plus model.path are all verl needs
    to resume, so the config.json write is non-essential."""
    if getattr(config_cls, "_verl_save_guarded", False):
        return
    _orig_save = config_cls.save_pretrained

    def _save_pretrained(self, *args, **kwargs):
        try:
            return _orig_save(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "verl_omni.qwen3_tts: skipped composite config save_pretrained (non-fatal: %s); "
                "model and optim shards are unaffected.",
                e,
            )

    config_cls.save_pretrained = _save_pretrained
    config_cls._verl_save_guarded = True


def _unwrap_non_tensor(v):
    """raw_prompt rides as a 0-d object array; unwrap to the underlying messages list."""
    if hasattr(v, "item") and not isinstance(v, list | tuple):
        try:
            return v.item()
        except ValueError:
            return v
    return v


def patch_agent_loop_multi_modal_inputs() -> None:
    """Inject the per-sample talker inputs into multi_modal_inputs so they reach the actor
    forward as kwargs. Response data is left-aligned; the FSDP engine re-packs input_ids."""
    try:
        from verl.experimental.agent_loop import agent_loop as _al
    except Exception as e:  # noqa: BLE001
        logger.warning("verl_omni.qwen3_tts: agent_loop patch skipped (%s).", e)
        return

    _orig = _al.AgentLoopWorker._compute_multi_modal_inputs

    def _patched(self, output, input_ids):
        mmi = _orig(self, output, input_ids)
        ef = getattr(output, "extra_fields", None) or {}
        codes = ef.get("tts_audio_codes")
        if codes is None:
            return mmi  # not a TTS rollout

        raw_prompt = _unwrap_non_tensor(ef.get("raw_prompt"))
        text = raw_prompt[0]["content"] if raw_prompt else ""
        tok = self.processor or self.tokenizer
        text_ids = tok(text=build_assistant_text(text), return_tensors="pt", padding=True)["input_ids"]
        text_ids = text_ids[:, :-TEXT_PROMPT_TRAILER_TOKENS].reshape(-1)

        R = int(self.rollout_config.response_length)
        Ttext = int(self.rollout_config.prompt_length)
        codes = torch.as_tensor(codes, dtype=torch.long)
        rl = min(codes.shape[0], R)
        tl = min(text_ids.numel(), Ttext)

        audio_buf = torch.zeros(1, R, codes.shape[-1], dtype=torch.long)
        audio_buf[0, :rl] = codes[:rl]
        text_buf = torch.zeros(1, Ttext, dtype=torch.long)
        text_buf[0, :tl] = text_ids[:tl]

        mmi["tts_audio_codes"] = audio_buf
        mmi["tts_text_ids"] = text_buf
        mmi["response_len"] = torch.tensor([rl], dtype=torch.long)
        mmi["text_len"] = torch.tensor([tl], dtype=torch.long)
        return mmi

    _al.AgentLoopWorker._compute_multi_modal_inputs = _patched
    logger.info("verl_omni.qwen3_tts: patched AgentLoopWorker._compute_multi_modal_inputs.")


_TTS_PASSTHROUGH_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def patch_hf_tokenizer_for_qwen3_tts() -> None:
    """Install a passthrough chat template: Qwen3-TTS-Base has none but verl's dataset calls
    apply_chat_template on every prompt."""
    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original_hf_tokenizer = _vt.hf_tokenizer

    def _patched_hf_tokenizer(name_or_path, **kwargs):
        tok = _original_hf_tokenizer(name_or_path, **kwargs)
        if tok is not None and not getattr(tok, "chat_template", None):
            tok.chat_template = _TTS_PASSTHROUGH_CHAT_TEMPLATE
            logger.info("verl_omni.qwen3_tts: installed passthrough chat_template on %s.", type(tok).__name__)
        return tok

    _vt.hf_tokenizer = _patched_hf_tokenizer


def patch_verl_ref_manual_offload() -> None:
    """Make verl's forward_only engine (the ref policy) use manual param offload instead of
    FSDP-native CPUOffload. The talker forward assembles its input by indexing leaf embedding
    tables and running text_projection outside the FSDP-wrapped module, so FSDP-native CPUOffload
    leaves those params on CPU and compute_ref_log_prob dies with a cuda/cpu device mismatch. verl
    forces CPUOffload for every forward_only engine regardless of param_offload; this neutralizes
    that only for the ref config this recipe runs (forward_only with param_offload false), where
    manual offload loads all params to GPU before the forward, the path the actor already takes."""
    try:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    except Exception as e:  # noqa: BLE001
        logger.warning("verl_omni.qwen3_tts: ref-offload patch skipped (%s).", e)
        return
    if getattr(FSDPEngine, "_verl_tts_ref_offload_patched", False):
        return

    def _run_as_manual_offload(engine, call):
        ec = engine.engine_config
        if not (getattr(ec, "forward_only", False) and not getattr(ec, "param_offload", True)):
            return call()
        ec.forward_only = False  # skip the forced-CPUOffload / early-return branches
        try:
            return call()
        finally:
            ec.forward_only = True

    _orig_build = FSDPEngine._build_fsdp_module
    _orig_to = FSDPEngine.to

    def _build_fsdp_module(self, module):
        return _run_as_manual_offload(self, lambda: _orig_build(self, module))

    def _to(self, device, model=True, optimizer=True, grad=True):
        return _run_as_manual_offload(self, lambda: _orig_to(self, device, model=model, optimizer=optimizer, grad=grad))

    FSDPEngine._build_fsdp_module = _build_fsdp_module
    FSDPEngine.to = _to
    FSDPEngine._verl_tts_ref_offload_patched = True
    logger.info("verl_omni.qwen3_tts: patched FSDPEngine for ref manual offload.")


def apply_qwen3_tts_patches() -> None:
    _register_qwen3_tts_automodel()
    patch_agent_loop_multi_modal_inputs()
    patch_hf_tokenizer_for_qwen3_tts()
    patch_verl_ref_manual_offload()


apply_qwen3_tts_patches()
