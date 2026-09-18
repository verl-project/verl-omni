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
"""CPU-only reward used by the Qwen3-TTS execution smoke test."""

import numpy as np


def compute_score(solution_audio, **kwargs):
    del kwargs
    waveform, sample_rate = solution_audio
    duration_s = np.asarray(waveform).size / sample_rate
    return {"score": float(duration_s), "duration_s": float(duration_s)}
