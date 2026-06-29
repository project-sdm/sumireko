#!/usr/bin/env bash
set -euo pipefail

curl -X POST -F "file=@$1" http://localhost:8000/audio/search-pgvector
