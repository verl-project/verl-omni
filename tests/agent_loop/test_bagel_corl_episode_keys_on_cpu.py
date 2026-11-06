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
"""Episode seed and artifact-key collision regressions for the live worker."""

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.agent_loop import episode_artifact_keys, episode_seed_index


def test_two_tasks_at_the_same_sibling_index_do_not_collide():
    """``session_id`` is the rollout *sibling* index, not the sample identity."""
    a_path, a_seed = episode_artifact_keys(dataset_task_uid="task_a", session_id=0, global_steps=60)
    b_path, b_seed = episode_artifact_keys(dataset_task_uid="task_b", session_id=0, global_steps=60)
    sibling_path, sibling_seed = episode_artifact_keys(dataset_task_uid="task_a", session_id=1, global_steps=60)

    assert a_path != b_path, "two tasks at sibling 0 must not share a trajectory folder"
    assert a_seed != b_seed, "two tasks at sibling 0 must not share a noise draw"
    assert a_path != sibling_path and a_seed != sibling_seed
    # ``sample_index`` carries the dataset sample, ``session_id`` goes to ``rollout_n``.
    assert a_path == "step_000060/sample_task_a.01"
    assert sibling_path == "step_000060/sample_task_a.02"


def test_episode_seed_index_is_process_stable_not_salted():
    """Replay has to redraw the same noise, so the index cannot use salted ``hash``."""
    import zlib

    assert episode_seed_index("uid:0") == episode_seed_index("uid:0")
    assert episode_seed_index("uid:0") == zlib.crc32(b"uid:0")
    assert episode_seed_index("uid:0") != episode_seed_index("uid:1")
