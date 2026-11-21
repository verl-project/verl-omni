#!/usr/bin/env bash
# From nothing to a closed session in six commands. Set the variables below, then run.
# The adapter lives at <skill>/adapters/session.py; the session directory is ./sessions/$ID.
set -euo pipefail

SESSION_TOOL="python3 /path/to/.agents/skills/experiment-campaign/adapters/session.py"
ID="${ID:-20261121-qwen3omni-step-time}"
DIR="${DIR:-./sessions/$ID}"
LAUNCHER="${LAUNCHER:-bash /path/to/your_launcher.sh}"
EXTRACT="${EXTRACT:-bash /path/to/extract_metrics.sh}"   # prints one number on stdout
TIMEOUT="${TIMEOUT:-25m}"
MISSION="${MISSION:-metric=timing_s/step,direction=lower,target=150}"

# 1. Mission and budget. direction has no default; repeats are what make the floor real.
$SESSION_TOOL init --dir "$DIR" --id "$ID" \
    --mission "$MISSION" \
    --max-rounds 12 --round-timeout "$TIMEOUT" \
    --launcher "$LAUNCHER" --extractor "$EXTRACT"

# 2. PRELUDE: anchor the baseline twice or three times (one round each).
for _ in 1 2 3; do
    ROUND=$($SESSION_TOOL start --dir "$DIR" --phase prelude --change none \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["round"])')
    timeout "$TIMEOUT" $LAUNCHER > "$DIR/rounds/$ROUND/artifacts/run.log" 2>&1
    VALUE=$($EXTRACT "$DIR/rounds/$ROUND/artifacts/run.log")
    $SESSION_TOOL result  --dir "$DIR" --round "$ROUND" --value "$VALUE" --why measured
    $SESSION_TOOL verdict --dir "$DIR" --round "$ROUND" --verdict baseline --why measured
done

# 3. OPTIMIZATION: one change per round; keep or revert on the re-measured number.
CHANGE="actor_rollout_ref.rollout.max_num_seqs=128"
ROUND=$($SESSION_TOOL start --dir "$DIR" --phase optimization --change "$CHANGE" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["round"])')
timeout "$TIMEOUT" $LAUNCHER "$CHANGE" > "$DIR/rounds/$ROUND/artifacts/run.log" 2>&1
VALUE=$($EXTRACT "$DIR/rounds/$ROUND/artifacts/run.log")
NOISE=$(python3 -c "import json; print(json.load(open('$DIR/state.json'))['noise_floor'])")
$SESSION_TOOL result  --dir "$DIR" --round "$ROUND" --value "$VALUE" --noise "$NOISE" --why improved
# Compare $VALUE with state.json's anchor/best, then record the verdict:
$SESSION_TOOL verdict --dir "$DIR" --round "$ROUND" --verdict keep --notes "why this should help"

# 4. …more optimization rounds and a sweep, then close and write the knowledge base.
$SESSION_TOOL close --dir "$DIR" --reason budget --detail "12 rounds used"
