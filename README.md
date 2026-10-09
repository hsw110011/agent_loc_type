# 卫星图定位分析 Agent

唯一图入口是 [`vlm/langgraph_main.py`](vlm/langgraph_main.py)，唯一启动脚本是 [`vlm/run_agent.sh`](vlm/run_agent.sh)。所有依赖合并为 [`vlm/requirements.txt`](vlm/requirements.txt)，Agent、PathFinder 和 INT8 VLM 共用项目 `.venv-vllm` 环境。

默认运行 Node 1 `read_picture` 和 Node 2 `extract_roads`；前两个节点不加载 VLM。运行日志按时间命名保存到 `vlm/log/`，任务产物、trace 和节点间 state 快照保存到 `vlm/output/<task_id>/`。

```bash
bash vlm/run_agent.sh --image /absolute/path/input.tif --primary-mode GPS
```

详细说明见 [`vlm/README.md`](vlm/README.md)。`PathFinder/` 保留第三方模型代码与权重；Node 3 的 `candidate_area_sys` 已接通本机 API Structured Outputs；使用 `--stop-after visualization` 可执行 Node 1～4。Node 5 留待后续处理。中间请求、响应、配置快照与 ROI 保存在 `vlm/runtime/<task_id>/`。
