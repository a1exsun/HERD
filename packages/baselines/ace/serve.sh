#!/usr/bin/env bash
# The OpenAI-compatible server every ACE role calls (ACE_LOCAL_BASE_URL=http://127.0.0.1:8000/v1).
#
#   serve.sh /path/to/qwen3.5-4b-checkpoint      (vllm==0.31.0; run.py checks the version)
#
# --max-model-len 20480 = the 16,384-token input ceiling + the 4,096 tokens every ACE call requests, so
# the server rejects (HTTP 400) any call whose prompt exceeds the ceiling. Decoding parameters come from
# the requests (upstream ACE: temperature 0). Thinking is disabled for every request by the server's
# default chat-template arguments, so upstream's requests need no change; run.py checks it via /tokenize.
set -euo pipefail
checkpoint="$(cd "$1" && pwd -P)"
exec vllm serve "$checkpoint" \
  --served-model-name Qwen/Qwen3.5-4B \
  --dtype bfloat16 \
  --max-model-len 20480 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --seed 42 \
  --host 127.0.0.1 --port 8000
