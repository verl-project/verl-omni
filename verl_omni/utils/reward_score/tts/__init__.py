# Copyright 2026 Gulp AI Inc and/or its affiliates
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

"""TTS reward functions, one module per model backend.

Each module owns its model as a lazy singleton and exposes compute_score, so rewards compose
via MultiAudioRewardManager's reward.reward_functions entries and each model is configured
independently. compute_score below is the all-in-one composition for the plain
AudioRewardManager default path.
"""


def compute_score(
    solution_audio,
    ground_truth: str,
    extra_info: dict = None,
    weights: dict = None,
    **kwargs,
) -> dict:
    """Weighted sum of the asr, sim, emo and stab dimensions (GLM-TTS reward forms).

    Every dimension is also returned so per-dimension advantage estimators (gdpo) can normalize
    them independently. Unscored dimensions get 0.0, below any scored clip. Per-model kwargs
    (whisper_model, spk_model, emo_model, ...) pass through to the underlying modules.

    Returns:
        dict: {"score", "asr", "sim", "emo", "stab", plus raw metrics}.
    """
    from verl_omni.utils.reward_score.tts import asr_reward, emo_reward, spk_sim_reward

    weights = {**{"asr": 1.0, "sim": 1.0, "emo": 1.0, "stab": 1.0}, **(weights or {})}
    asr = asr_reward.compute_score(solution_audio, ground_truth, extra_info, **kwargs)
    sim = spk_sim_reward.compute_score(solution_audio, extra_info, **kwargs)
    emo = emo_reward.compute_score(solution_audio, extra_info, **kwargs)

    score = (
        weights["asr"] * asr["score"]
        + weights["sim"] * sim["score"]
        + weights["emo"] * emo["score"]
        + weights["stab"] * asr["stab"]
    )
    return {
        "score": score,
        "asr": asr["score"],
        "sim": sim["score"],
        "emo": emo["score"],
        "stab": asr["stab"],
        "cer": asr["cer"],
        "cos": sim["cos"],
        "p_neutral": emo["p_neutral"],
        "synth_ok": asr["synth_ok"],
        "truncated": asr["truncated"],
        "repeated": asr["repeated"],
    }


__all__ = ["compute_score"]
