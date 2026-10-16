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
"""CPU tests for the class-based PickScore scorer and its registry."""

import pytest
import torch

from verl_omni.pipelines.rollout_media import ArtifactContractError
from verl_omni.utils.reward_score.scorers import pickscore as port
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import get_scorer_cls, peek_registered_scorers


class _FakeWeights:
    def __init__(self, device):
        self.device = torch.device(device)
        self.moves = []

    def to(self, device):
        self.moves.append(torch.device(device))
        self.device = torch.device(device)
        return self


class _FakeInferencer:
    """Deterministic 2-D embeddings: "cat" prompt and cat-like image both map to [1, 0]."""

    logit_scale = torch.tensor(100.0)

    def __init__(self, *, device, dtype=None, model_path=None, processor_path=None):
        self.device = torch.device(device)
        self.model = _FakeWeights(device)

    def infer(self, prompts, images):
        text = torch.tensor([[1.0, 0.0] if prompt == "cat" else [0.0, 1.0] for prompt in prompts])
        image = torch.tensor([[1.0, 0.0] if self._is_cat_image(item) else [0.0, 1.0] for item in images])
        return {"text_embeddings": text, "image_embeddings": image, "logit_scale": self.logit_scale}

    @staticmethod
    def _is_cat_image(image):
        return sum(image.convert("L").getdata()) % 2 == 0


def _image(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, 8, 8), generator=generator, dtype=torch.uint8)


def _request(prompt: str, seed: int, **extra) -> dict:
    request = {
        "solution_image": _image(seed),
        "ground_truth": prompt,
        "extra_info": {"media_kind": "image"},
    }
    request.update(extra)
    return request


def _expected_score(request: dict, divisor: float = 26.0) -> float:
    """Fake embeddings give cos=1 exactly when prompt and image agree on cat-ness."""
    agrees = (request["ground_truth"] == "cat") == _is_cat(request)
    return 100.0 / divisor if agrees else 0.0


def _is_cat(request: dict) -> bool:
    import verl_omni.utils.reward_score.pickscore_reward as frozen

    image = frozen._to_pil_hwc(request["solution_image"])
    return _FakeInferencer._is_cat_image(image)


@pytest.fixture
def fake_inferencer(monkeypatch):
    created = []

    def build(**kwargs):
        inferencer = _FakeInferencer(**kwargs)
        created.append(inferencer)
        return inferencer

    monkeypatch.setattr(port._frozen, "_PickScoreInferencer", build)
    return created


def test_registry_resolves_pickscore_by_name_and_import_spec():
    assert "pickscore" in peek_registered_scorers()
    assert get_scorer_cls("pickscore") is port.PickScoreScorer
    ref = "pkg://verl_omni.utils.reward_score.scorers.pickscore:PickScoreScorer"
    assert get_scorer_cls(ref) is port.PickScoreScorer


def test_registry_unknown_name_lists_registered_names():
    with pytest.raises(ValueError, match="Unknown reward scorer 'no-such-scorer'") as excinfo:
        get_scorer_cls("no-such-scorer")
    assert "Registered names:" in str(excinfo.value)
    assert "pickscore" in str(excinfo.value)


def test_registry_duplicate_registration_conflicts_but_same_class_is_idempotent():
    from verl_omni.utils.reward_score.scorers.registry import register_scorer

    with pytest.raises(ValueError, match="has already been registered"):

        @register_scorer("pickscore")
        class ConflictingScorer(RewardScorer):
            def score_batch(self, batch):
                return [{"score": 0.0} for _ in batch]

    # Re-registering the identical class (module re-import) is a tolerated no-op.
    assert register_scorer("pickscore")(get_scorer_cls("pickscore")) is get_scorer_cls("pickscore")
    assert get_scorer_cls("pickscore") is port.PickScoreScorer


def test_registry_rejects_non_scorer_reference():
    with pytest.raises(ValueError, match="does not subclass RewardScorer"):
        get_scorer_cls("verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel")


def test_registry_resolves_file_path_reference(tmp_path):
    scorer_file = tmp_path / "out_of_tree_scorer.py"
    scorer_file.write_text(
        "from verl_omni.utils.reward_score.scorers.base import RewardScorer\n"
        "class OutOfTreeScorer(RewardScorer):\n"
        "    def score_batch(self, batch):\n"
        "        return [{'score': 0.0} for _ in batch]\n"
    )
    cls = get_scorer_cls(f"{scorer_file}:OutOfTreeScorer")
    assert issubclass(cls, RewardScorer)


def test_registered_score_batch_scores_on_cpu(fake_inferencer):
    scorer = get_scorer_cls("pickscore")()
    scorer.load(torch.device("cpu"))

    requests = [_request("cat", seed=0), _request("cat", seed=1), _request("dog", seed=2)]
    results = scorer.score_batch(requests)

    assert len(results) == len(requests)
    for request, result in zip(requests, results, strict=True):
        expected = _expected_score(request)
        assert result["score"] == pytest.approx(expected)
        assert result["pickscore_raw"] == pytest.approx(expected)


def test_score_batch_isolates_per_request_failures_and_keeps_request_order(fake_inferencer):
    scorer = port.PickScoreScorer()
    scorer.load(torch.device("cpu"))

    good = _request("cat", seed=3)
    contract_violation = {"solution_image": None, "ground_truth": "cat", "extra_info": {}}
    custom_divisor = _request("cat", seed=4, score_divisor=13.0)
    results = scorer.score_batch([good, contract_violation, custom_divisor])

    assert results[0]["score"] == pytest.approx(_expected_score(good))
    assert isinstance(results[1], ArtifactContractError)
    assert results[2]["score"] == pytest.approx(_expected_score(custom_divisor, divisor=13.0))


def test_score_batch_rejects_zero_divisor_per_request(fake_inferencer):
    scorer = port.PickScoreScorer()
    scorer.load(torch.device("cpu"))
    results = scorer.score_batch([_request("cat", seed=0, score_divisor=0.0)])
    assert isinstance(results[0], ValueError)


@pytest.mark.asyncio
async def test_sleep_wake_close_choreography(fake_inferencer):
    scorer = port.PickScoreScorer()
    scorer.load(torch.device("cpu"))
    inferencer = fake_inferencer[0]
    weights = inferencer.model

    scorer.sleep()
    assert weights.moves == [torch.device("cpu")]
    assert inferencer.device == torch.device("cpu")

    scorer.wake_up()
    assert weights.moves[-1] == scorer.device
    assert inferencer.device == scorer.device

    scorer.close()
    assert scorer.model is None
    assert scorer._inferencer is None


def test_hooks_before_load_fail_fast():
    scorer = port.PickScoreScorer()
    with pytest.raises(RuntimeError, match="load"):
        scorer.score_batch([_request("cat", seed=0)])


@pytest.mark.asyncio
async def test_glue_function_submits_one_request_dict_to_the_executor():
    class _FakeExecutor:
        def __init__(self):
            self.requests = []

        async def score(self, request):
            self.requests.append(request)
            return {"score": 0.5, "pickscore_raw": 0.5}

    executor = _FakeExecutor()
    extra_info = {"media_kind": "image"}
    result = await port.compute_score_pickscore_batched(
        data_source="prompt Britt",
        solution_image=_image(0),
        ground_truth="cat",
        extra_info=extra_info,
        reward_model=executor,
    )
    assert result == {"score": 0.5, "pickscore_raw": 0.5}
    (submitted,) = executor.requests
    assert submitted["data_source"] == "prompt Britt"
    assert submitted["solution_image"].shape == _image(0).shape
    assert submitted["ground_truth"] == "cat"
    assert submitted["extra_info"] == extra_info
    assert submitted["score_divisor"] == 26.0
