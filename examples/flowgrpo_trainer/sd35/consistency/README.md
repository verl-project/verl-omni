# Public SD3.5 first-forward consistency capture

Last updated: 11/09/2026

This recipe collects a real SD3.5 Medium rollout and PPO FSDP actor pair for
`scripts/replay_sd35_consistency.py`. It needs a local complete SD3.5 Medium
checkpoint at revision `b940f670f0eda2d07fbb75229e779da1ad11eb80`, a
compatible installed vLLM Omni and MSProbe runtime, and one visible 80 GB GPU
for the default capture.
The prior functional probe used `mindstudio-probe==26.2.0` (Python module
`msprobe`); this new recipe has not yet been validated in that runtime.
The five pipeline components stored as FP16 total about 16.6 GB; loading all
components in FP32 needs about 33 GB before activations. A 24 GB GPU is not a
supported target for the default mode. An explicit DiT-only layerwise rollout
offload mode is available for narrower GPUs; the actor runs separately. No
download or package installation is performed.

All paths below are supplied by the caller. Use an absolute deployed source
checkout/snapshot, an existing model directory, a fresh run name, and an output
parent with sufficient space. `PYTHON` names the environment's Python binary.
A Git checkout can derive `SOURCE_HEAD`; a deployed snapshot must provide a
40-character `SOURCE_HEAD` explicitly. The manifest is a producer-declared
byte inventory, not an external source attestation.

```bash
export SOURCE_ROOT=/absolute/source PYTHON=/absolute/python
export SOURCE_MANIFEST=/absolute/source-manifest.json
# For a deployed snapshot: export SOURCE_HEAD=<40-character-Git-revision>
bash "$SOURCE_ROOT/examples/flowgrpo_trainer/sd35/consistency/run.sh" manifest
# Retain the printed manifest SHA256 separately, then supply it:
export SOURCE_MANIFEST_SHA256=<printed-sha256>
export SOURCE_HEAD=<declared-or-git-head>
export MODEL_DIR=/absolute/local/model OUTPUT_ROOT=/absolute/capture-parent
export RUN_ID=sd35-public-001 PIN_FILE=/absolute/outside-capture/producer-pins.sha256
# Optional: set before both admit and capture; default is none.
# export SD35_ROLLOUT_OFFLOAD=dit-layerwise
bash "$SOURCE_ROOT/examples/flowgrpo_trainer/sd35/consistency/run.sh" admit
# On the GPU host, using the same source, model, manifest, run ID, and output root:
bash "$SOURCE_ROOT/examples/flowgrpo_trainer/sd35/consistency/run.sh" capture
```

`admit` masks CUDA visibility and checks API/config/scheduler contracts without
CUDA initialization. `capture` verifies the admission token against the same
source/model/config identity, including rollout offload mode, then runs rollout
and actor in separate GPU processes. Keep `SD35_ROLLOUT_OFFLOAD` identical for
`admit` and `capture`; changing it requires a fresh admission and run root.
`none` passes no offload configuration. `dit-layerwise` requests the pinned
vLLM Omni DiT layerwise backend, verifies that it is enabled, and prints rollout
load and after each request, sampling CUDA peak allocated, peak reserved, and
current free bytes. It captures only the first sample's first actual
transformer forward per role with MSProbe. The saved actor batch and comparison
contain two prompts (`A red cube on a white table.`, `A blue sphere on a white
table.`), seeds 1234/1235 and two SDE steps. Fixed settings are 256 × 256,
four denoising steps, FP32, native actor attention, TORCH_SDPA rollout,
classifier-free guidance 0, microbatch 1, noise 0.7. No training update occurs.

A FP32 run of this recipe on one RTX 3090 with DiT-only layerwise offload,
based on main `d65b2017`, passed capture and replay. The highest sampled CUDA
peak allocation was 22.94 GiB, with 263 MiB free after each request. This
measurement applies to the fixed inputs and runtime used for that run.

The capture command writes `PIN_FILE` outside the run and prints its SHA256.
A completed numerical mismatch still writes the eight pins and exits with status
10 so the public CPU reader can diagnose it. An incomplete rollout, actor, or
MSProbe failure exits without generating a new pin file.
Retain that digest independently before replay. Recomputing it from the same
untrusted pin file at replay time supplies no independent byte evidence.

```bash
export TRUSTED_PIN_SHA256=<independently-retained-pin-file-digest>
export REPLAY_OUTPUT=/absolute/fresh-replay-output
bash "$SOURCE_ROOT/examples/flowgrpo_trainer/sd35/consistency/run.sh" replay
```

The reader verifies the eight producer pins and compares the full 2 × 2
log-probability tensors. `audit.json` reports the overall replay status.
`report.json` covers the public checker's validation of the MSProbe dumps and
log probabilities for row 0, step 0. A numerical mismatch anywhere in the
saved tensors or a malformed capture causes a nonzero exit. Partial artifacts
stay in place for diagnosis. The transformer checkpoint byte SHA256 is checked
against this recipe's fixed digest before either GPU role. The remaining model
component files are inventoried by path and size, and their configs are hashed;
that does not independently attest those weights. Real execution still needs
CPU admission and GPU validation on the target runtime.
