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

"""CPU coverage for DVReward named-image selection and HTTP scoring."""

import asyncio
import base64
import math
from io import BytesIO

import numpy as np
import pytest
import torch
from aiohttp import web
from PIL import Image

from verl_omni.pipelines.rollout_media import ArtifactContractError, MediaArtifact, MediaSpec
from verl_omni.utils.reward_score.dvreward import _yes_probability, compute_score_dvreward


def image_info():
    return {"semantic_questions": ["Is there a cat?"], "quality_questions": ["Is it clear?"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [True, False])
async def test_scores_selected_image_with_latent_primary(named):
    pixels = torch.full((3, 8, 8), 73, dtype=torch.uint8)
    latent = torch.zeros(4, 16)
    info = image_info()
    if named:
        info.update(
            media_artifacts={
                "image_latent": MediaArtifact(MediaSpec("image", "latent", "LC"), latent),
                "unused_image": MediaArtifact(MediaSpec("image", "decoded", "CHW"), torch.zeros_like(pixels)),
                "chosen_image": MediaArtifact(MediaSpec("image", "decoded", "CHW"), pixels),
            },
            primary_artifact="image_latent",
            preview_artifact="chosen_image",
        )
    else:
        info["media_kind"] = "image"
    seen = []

    async def handler(request):
        body = await request.json()
        encoded = body["messages"][1]["content"][0]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as image:
            np.testing.assert_array_equal(np.asarray(image), pixels.permute(1, 2, 0).numpy())
        seen.append(body)
        return web.json_response(
            {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "yes", "logprob": math.log(0.8)},
                                        {"token": "no", "logprob": math.log(0.2)},
                                    ]
                                }
                            ]
                        },
                        "message": {"content": "yes"},
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        result = await compute_score_dvreward(
            "test", latent if named else pixels, "cat", info, f"127.0.0.1:{port}", "judge"
        )
    finally:
        await runner.cleanup()
    assert result["score"] == pytest.approx(0.8)
    assert len(seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["absent", "none", "latent", "video", "hwc", "float_pixels", "unlabelled"])
async def test_invalid_selection_fails_before_http(case):
    pixels = torch.zeros(3, 8, 8, dtype=torch.uint8)
    preview = MediaArtifact(MediaSpec("image", "decoded", "CHW"), pixels)
    if case == "latent":
        preview = MediaArtifact(MediaSpec("image", "latent", "LC"), torch.zeros(4, 16))
    elif case == "video":
        preview = MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=24), pixels.unsqueeze(0))
    elif case == "hwc":
        preview = MediaArtifact(MediaSpec("image", "decoded", "HWC"), pixels.permute(1, 2, 0))
    elif case == "float_pixels":
        preview = MediaArtifact(MediaSpec("image", "decoded", "CHW"), pixels.float())
    info = image_info()
    if case != "unlabelled":
        info.update(
            media_artifacts={"preview": preview},
            preview_artifact=None if case == "none" else "missing" if case == "absent" else "preview",
        )
    with pytest.raises(ArtifactContractError):
        await asyncio.wait_for(compute_score_dvreward("test", pixels, "cat", info, "unreachable", "judge"), 1)


@pytest.mark.asyncio
async def test_empty_legacy_image_fails_before_http():
    info = {**image_info(), "media_kind": "image"}
    with pytest.raises(ArtifactContractError):
        await compute_score_dvreward(
            "test", torch.empty(3, 0, 8, dtype=torch.uint8), "cat", info, "unreachable", "judge"
        )


@pytest.mark.parametrize(
    "logprobs", [None, {}, {"content": None}, {"content": []}, {"content": [{}]}, {"content": [{"top_logprobs": None}]}]
)
@pytest.mark.parametrize("answer,expected", [("yes", 1.0), ("No.", 0.0)])
def test_missing_logprobs_falls_back_to_answer(logprobs, answer, expected):
    choice = {"message": {"content": answer}}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    else:
        choice["logprobs"] = None
    assert _yes_probability({"choices": [choice]}) == expected


def test_absent_logprobs_falls_back_to_answer():
    assert _yes_probability({"choices": [{"message": {"content": "yes"}}]}) == 1.0


def test_fallback_rejects_non_binary_answer():
    with pytest.raises(ValueError, match="yes or no"):
        _yes_probability({"choices": [{"logprobs": None, "message": {"content": "maybe"}}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_parent", [False, True])
async def test_pending_questions_finish_before_session_closes(monkeypatch, cancel_parent):
    from contextlib import asynccontextmanager

    from verl_omni.utils.reward_score import dvreward

    started = asyncio.Event()
    tasks = []
    finished_before_close = []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            finished_before_close.extend(task.done() for task in tasks)

        @asynccontextmanager
        async def post(self, url, json):
            tasks.append(asyncio.current_task())
            question = json["messages"][1]["content"][1]["text"]
            if question == "Is there a cat?" and not cancel_parent:
                await started.wait()
                raise ValueError("judge request failed")
            started.set()
            await asyncio.Event().wait()
            yield

    monkeypatch.setattr(dvreward.aiohttp, "ClientSession", lambda **kwargs: Session())
    parent = asyncio.create_task(
        compute_score_dvreward(
            "test",
            torch.zeros(3, 8, 8, dtype=torch.uint8),
            "cat",
            {**image_info(), "media_kind": "image"},
            "unused",
            "judge",
        )
    )
    try:
        if cancel_parent:
            await asyncio.wait_for(started.wait(), 2)
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await parent
        else:
            with pytest.raises(ValueError, match="judge request failed"):
                await asyncio.wait_for(parent, 2)
        assert len(tasks) == 2
        assert finished_before_close == [True, True]
        assert tasks[-1].cancelled()
    finally:
        for task in [parent, *tasks]:
            task.cancel()
        await asyncio.gather(parent, *tasks, return_exceptions=True)
