#!/bin/sh
# Start the Strata-DeepSeek web server with the config beside this script.
#   ./run.sh                 -> strata-ds4.json, port 8080
#   ./run.sh other.json      -> another config
#   PORT=9000 ./run.sh       -> another port
set -e
cd "$(dirname "$0")"
exec python3 -m serve.server --engine strata --config "${1:-strata-ds4.json}" --port "${PORT:-8080}" --open
