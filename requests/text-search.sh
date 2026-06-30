#!/usr/bin/env bash
set -euo pipefail


curl -G --data-urlencode "q=$1" "http://localhost:8000/text/search"
