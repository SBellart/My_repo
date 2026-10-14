#!/usr/bin/env bash
# One-command local demo: ingest a dataset (HuggingFace -> local embedder ->
# TinyDB) and run corrections end to end. Initial downloads require Internet;
# MiniLM needs no database server or credentials. See README.md for the options (variant,
# embedder backend, dataset, seed) to run other experiments.
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$HERE/.venv/bin/python" ]; then
    PYTHON="$HERE/.venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

DATASET="scifact"
VARIANT="both"
BACKEND="sentence_transformers"
MODEL=""
DB_DIR="$HERE/local_db_demo"
SEED=""
EPOCHS=30
ITERATIONS=30
N_CLUSTERS=3
API_KEY=""
API_BASE_URL=""

usage() {
    cat <<'EOF'
Usage: run_local_demo.sh [--dataset NAME] [--variant requete|cluster|both]
                          [--embedder-backend sentence_transformers|openai]
                          [--embedder-model MODEL] [--seed N]
                          [--epochs N] [--iterations N] [--n-clusters N]
                          [--api-key KEY] [--api-base-url URL]

Examples:
  ./run_local_demo.sh                              # both variants (default)
  ./run_local_demo.sh --variant requete            # global variant only
  ./run_local_demo.sh --variant cluster            # clustered variant only
  ./run_local_demo.sh --dataset nfcorpus
  ./run_local_demo.sh --embedder-backend openai --embedder-model text-embedding-3-small
  ./run_local_demo.sh --seed 7
  ./run_local_demo.sh --epochs 5 --iterations 3
  ./run_local_demo.sh --n-clusters 5
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --embedder-backend) BACKEND="$2"; shift 2 ;;
        --embedder-model) MODEL="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --n-clusters) N_CLUSTERS="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --api-base-url) API_BASE_URL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [ "$VARIANT" != "requete" ] && [ "$VARIANT" != "cluster" ] && [ "$VARIANT" != "both" ]; then
    echo "Invalid --variant: $VARIANT (expected requete, cluster, or both)" >&2
    exit 1
fi

if ! [[ "$N_CLUSTERS" =~ ^[0-9]+$ ]] || [ "$N_CLUSTERS" -lt 1 ]; then
    echo "Invalid --n-clusters: $N_CLUSTERS (expected positive integer)" >&2
    exit 1
fi

MODEL_ARGS=()
if [ -n "$MODEL" ]; then
    MODEL_ARGS=(--embedder-model "$MODEL")
fi
if [ -n "$API_KEY" ]; then
    MODEL_ARGS+=(--api-key "$API_KEY")
fi
if [ -n "$API_BASE_URL" ]; then
    MODEL_ARGS+=(--api-base-url "$API_BASE_URL")
fi

# Alias printed by direct_feed_local.py as "Wrote alias '...'"; reconstructed
# here so the demo can chain both steps without reparsing that output.
DEFAULT_MODEL_SLUG="sentence-transformers-all-minilm-l6-v2"
case "$BACKEND" in
    sentence_transformers) MODEL_SLUG="${MODEL:-sentence-transformers/all-MiniLM-L6-v2}" ;;
    openai) MODEL_SLUG="${MODEL:-text-embedding-3-small}" ;;
    *) echo "Unknown --embedder-backend: $BACKEND" >&2; exit 1 ;;
esac
SAFE_MODEL="$(echo "$MODEL_SLUG" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]/-/g' | sed 's/^-*//;s/-*$//')"
SAFE_BACKEND="$(echo "$BACKEND" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]/-/g')"
ALIAS="${DATASET}__emb__${SAFE_BACKEND}__${SAFE_MODEL}"

SEED_ARGS=()
if [ -n "$SEED" ]; then
    SEED_ARGS=(--seed "$SEED")
fi

run_correction() {
    local v="$1"
    local script="$HERE/code/0_math_approach/debruitage_evalue_${v}.py"
    local extra_args=()
    if [ "$v" = "cluster" ]; then
        extra_args=(--n-clusters "$N_CLUSTERS")
    fi

    echo
    echo "== Run the $v-variant correction against the local store =="
    PARASITIC_LOCAL_DB="$DB_DIR" $PYTHON "$script" \
        --dataset "mteb/$ALIAS" \
        --epochs "$EPOCHS" --iterations "$ITERATIONS" \
        --healthy-top-k 10 --error-buffer 10 \
        --tau-h 0.05 --tau-e 0.05 --eval-every 1 \
        --val-fraction 0.2 --test-fraction 0.2 \
        --gate-acceptance-strategy strict --search-strictness strict \
        "${extra_args[@]}" "${SEED_ARGS[@]}"
}

echo "== Step 1: ingest '$DATASET' from HuggingFace into a local TinyDB store (backend=$BACKEND) =="
$PYTHON "$HERE/code/0_math_approach/direct_feed_local.py" \
    --dataset "$DATASET" \
    --embedder-backend "$BACKEND" \
    "${MODEL_ARGS[@]}" \
    --db-dir "$DB_DIR"

if [ "$VARIANT" = "both" ]; then
    run_correction requete
    run_correction cluster
else
    run_correction "$VARIANT"
fi

echo
echo "Demo complete ($EPOCHS epochs / $ITERATIONS iterations, variant=$VARIANT)."
echo "Pass --epochs 5 --iterations 3 for a reduced-budget smoke test."
