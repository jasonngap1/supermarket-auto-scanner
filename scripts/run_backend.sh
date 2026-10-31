#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$(pwd)/../data/hf_cache}"
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 "$@"
