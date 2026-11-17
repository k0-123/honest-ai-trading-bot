#!/usr/bin/env bash
# Daily paper run. Schedule at ~16:30 IST Mon-Fri. Keys come from ~/.nsebot.env (chmod 600), never from git.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f "$HOME/.nsebot.env" ] && set -a && . "$HOME/.nsebot.env" && set +a
mkdir -p logs
exec .venv/bin/nsebot daily >> "logs/daily_$(date +%Y-%m).log" 2>&1
