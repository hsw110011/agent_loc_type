#!/usr/bin/env bash
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$MODULE_DIR")"
PYTHON="${PROJECT_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
[[ "$PYTHON" = /* ]] || PYTHON="$PROJECT_DIR/$PYTHON"
[[ -x "$PYTHON" ]] || { echo "项目 Python 不存在：$PYTHON" >&2; exit 1; }

BASE_LIB="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.base_prefix) / "lib")')"
if [[ -f "$BASE_LIB/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="$BASE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false PYTHONUNBUFFERED=1
exec "$PYTHON" "$MODULE_DIR/map_preprocess/main.py" "$@"
