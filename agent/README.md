# Agent 编排与本地 VLM

`map_preprocess/graph.py` 是整图预处理 LangGraph（global_blocks，Node 1-5），配置、state、节点、布块与出图均在该子包中。神经网络由 `percection_tool` 提供，预处理没有真实 VLM 调用。

`vlm/` 保留本地 VLM 的图片编码、HTTP 推理、EXL3 服务（`exl3_server.py`）延迟启动与回收、JSON Schema 约束和 thinking 预算。未来的 `parallel_vlm.py` 定位评估仍需补齐业务配置。

统一启动：`bash agent/run_agent.sh --image input_geo_img/kitti_0930_0033.tif --config agent/map_preprocess/pipeline.yaml`。依赖来自根目录 `requirements.txt`；VLM 模型项目在根目录 `qwen3827b-exl3-dflash2/`。

`paths.py` 提供不依赖模型的路径和 JSON 工具，并将迁移前 manifest 中的路径映射到现有位置。新产物位于根目录 `output/`。
