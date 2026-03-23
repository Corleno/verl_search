#!/usr/bin/env bash
# Requires env SERPAPI_API_KEY (never commit the key here). Examples:
#   - add: export SERPAPI_API_KEY='...'  to ~/.bashrc  then: source ~/.bashrc
#   - or one-shot: SERPAPI_API_KEY='...' bash serp_search_launch.sh

search_url="https://serpapi.com/search"

# check if the serp api key is set
if [ -z "$SERPAPI_API_KEY" ]; then
    echo "SERPAPI_API_KEY is not set"
    exit 1
fi
serp_api_key="${SERPAPI_API_KEY:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/serp_search_server.py" --search_url "$search_url" --topk 3 --serp_api_key "$serp_api_key"
