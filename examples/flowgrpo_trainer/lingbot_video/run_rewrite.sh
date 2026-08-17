#!/usr/bin/env bash
# LingBot prompt rewrite launcher.
set -euo pipefail

ROOT=${ROOT:-${HOME}}
PY=${PY:-python3}
REWRITER_DIR=${REWRITER_DIR:-$ROOT/lingbot-video/rewriter}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

INPUT=${INPUT:-$ROOT/data/lingbot_video/prompts_clean.txt}
OUTPUT=${OUTPUT:-$ROOT/data/lingbot_video/rewritten_vllm/all.jsonl}
LOGDIR=${LOGDIR:-$ROOT/logs/rewrite}
BACKEND=${BACKEND:-vllm}
DURATION=${DURATION:-5}
MODE=${MODE:-t2v}

BASE_URL=${BASE_URL:-http://127.0.0.1:8137}
CONCURRENCY=${CONCURRENCY:-256}

export REWRITER_BASE_MODEL=${REWRITER_BASE_MODEL:-$ROOT/models/Qwen3.6-27B}
export REWRITER_ADAPTER=${REWRITER_ADAPTER:-$ROOT/models/lingbot-video-rewriter-lora}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
OUTDIR=${OUTDIR:-$(dirname "$OUTPUT")}

mkdir -p "$(dirname "$OUTPUT")" "$LOGDIR"

merge_into_output() {
    "$PY" - "$OUTPUT" "$@" <<'PY'
import json
import os
import sys
import tempfile

output, *sources = sys.argv[1:]
directory = os.path.dirname(output) or "."
seen = set()
kept = skipped = malformed = 0
fd, temporary = tempfile.mkstemp(prefix=".rewrite-merge-", suffix=".jsonl", dir=directory, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as destination:
        for source in [output, *sources]:
            if not os.path.exists(source):
                continue
            with open(source, encoding="utf-8") as input_file:
                for line in input_file:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        prompt = record["prompt_raw"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        malformed += 1
                        continue
                    if prompt in seen:
                        skipped += 1
                        continue
                    seen.add(prompt)
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
    os.replace(temporary, output)
except BaseException:
    os.unlink(temporary)
    raise
print(
    f"Merged {len(sources)} shard(s) into {output}: "
    f"kept={kept}, duplicates_removed={skipped}, malformed_skipped={malformed}"
)
PY
}

CARRY=${CARRY:-$ROOT/data/lingbot_video/rewritten}
if [ -d "$CARRY" ] && ls "$CARRY"/shard_*.jsonl >/dev/null 2>&1; then
    merge_into_output "$CARRY"/shard_*.jsonl
fi

if [ "$BACKEND" = "vllm" ]; then
    echo "Rewriting $INPUT -> $OUTPUT via $BASE_URL (vllm, concurrency $CONCURRENCY)"
    exec "$PY" "$SCRIPT_DIR/rewrite_prompts.py" \
        --backend vllm \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --rewriter-dir "$REWRITER_DIR" \
        --base-url "$BASE_URL" \
        --mode "$MODE" \
        --duration "$DURATION" \
        --concurrency "$CONCURRENCY"
elif [ "$BACKEND" = "transformers" ]; then
    IFS=',' read -ra GARR <<< "$GPUS"
    N=${#GARR[@]}
    echo "Rewriting $INPUT across $N GPU(s): $GPUS (transformers)"
    echo "Base=$REWRITER_BASE_MODEL  Adapter=$REWRITER_ADAPTER"
    pids=()
    for i in "${!GARR[@]}"; do
        g=${GARR[$i]}
        CUDA_VISIBLE_DEVICES="$g" nohup "$PY" "$SCRIPT_DIR/rewrite_prompts.py" \
            --backend transformers \
            --input "$INPUT" \
            --output "$OUTDIR/shard_${i}.jsonl" \
            --rewriter-dir "$REWRITER_DIR" \
            --shard "$i" \
            --num-shards "$N" \
            --mode "$MODE" \
            --duration "$DURATION" \
            > "$LOGDIR/shard_${i}.log" 2>&1 &
        pids+=("$!")
        echo "  shard $i on GPU $g -> PID ${pids[-1]}  (log: $LOGDIR/shard_${i}.log)"
    done
    echo "Launched ${#pids[@]} shards; waiting..."
    failed=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "ERROR: shard $i (PID ${pids[$i]}) failed — see $LOGDIR/shard_${i}.log" >&2
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "One or more shards failed; not merging a partial dataset." >&2
        exit 1
    fi
    merge_into_output "$OUTDIR"/shard_*.jsonl
    echo "All shards finished and merged into $OUTPUT"
else
    echo "Unknown BACKEND='$BACKEND' (expected 'vllm' or 'transformers')" >&2
    exit 2
fi
