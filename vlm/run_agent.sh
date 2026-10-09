#!/usr/bin/env bash
# 唯一启动脚本：Agent 与首次推理时启动的 VLM 共用一个 Python 环境。
set -euo pipefail
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_PYTHON:-$MODULE_DIR/../.venv-vllm/bin/python}"
[[ "$PYTHON" = /* ]] || PYTHON="$MODULE_DIR/../$PYTHON"
[[ -x "$PYTHON" ]] || { echo "Agent Python 不存在: $PYTHON" >&2; exit 1; }
BASE_LIB="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.base_prefix) / "lib")')"
if [[ -f "$BASE_LIB/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="$BASE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# 默认不上传 LangSmith 跟踪；本地 VLM 仅访问本机服务。
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
export PROJECT_PYTHON="$PYTHON"
export PYTHONUNBUFFERED=1
LOG_DIR="$MODULE_DIR/log"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)_agent.log"
"$PYTHON" "$MODULE_DIR/langgraph_main.py" "$@" 2>&1 | tee "$LOG_PATH"
