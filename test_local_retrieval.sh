#!/usr/bin/env bash
# Integration tests for the local dense retrieval server started by start_local_retrieval.sh
# Prerequisites: server listening (default http://127.0.0.1:8000), curl, python3.
#
# Usage:
#   ./test_local_retrieval.sh
#   BASE_URL=http://127.0.0.1:9000 ./test_local_retrieval.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
MAX_TIME="${MAX_TIME:-300}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required (for JSON validation)"

retrieve_json() {
  local payload="$1"
  curl -sS \
    --connect-timeout "${CONNECT_TIMEOUT}" \
    --max-time "${MAX_TIME}" \
    -X POST "${BASE_URL%/}/retrieve" \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    -w "\n%{http_code}"
}

check_http_and_json() {
  local name="$1"
  local payload="$2"
  local validator="$3"

  echo "==> ${name}"
  local raw
  raw="$(retrieve_json "${payload}")" || fail "curl failed for: ${name}"

  local http_code
  http_code="$(printf '%s' "${raw}" | tail -n1)"
  local body
  body="$(printf '%s' "${raw}" | sed '$d')"

  [[ "${http_code}" == "200" ]] || fail "${name}: expected HTTP 200, got ${http_code}. Body: ${body}"

  printf '%s' "${body}" | python3 -c "${validator}" || fail "${name}: response validation failed"
  echo "    OK"
}

echo "Local retrieval server tests against ${BASE_URL}/retrieve"
echo

# Single query, documents only (matches QueryRequest.return_scores=false)
check_http_and_json \
  "Single query without scores" \
  '{"queries":["What is supervised learning?"],"topk":2,"return_scores":false}' \
  'import json,sys
d=json.load(sys.stdin)
assert "result" in d and isinstance(d["result"], list) and len(d["result"])==1
q0=d["result"][0]
assert isinstance(q0, list) and len(q0)>=1 and len(q0)<=2
assert isinstance(q0[0], dict)
'

# Batch + scores (matches QueryRequest.return_scores=true)
check_http_and_json \
  "Batch queries with scores" \
  '{"queries":["capital of France","photosynthesis definition"],"topk":3,"return_scores":true}' \
  'import json,sys
d=json.load(sys.stdin)
assert "result" in d and isinstance(d["result"], list) and len(d["result"])==2
for q in d["result"]:
  assert isinstance(q, list) and 1 <= len(q) <= 3
  for hit in q:
    assert isinstance(hit, dict)
    assert "document" in hit and "score" in hit
    assert isinstance(hit["score"], (int, float))
'

echo
echo "All retrieval server checks passed."
