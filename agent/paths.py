"""项目路径与轻量 JSON；感知工具无需导入 LangGraph 或 VLM 即可使用。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 兼容迁移前 manifest/config 中的路径；历史记录不改写，返回新的实际位置。
LEGACY_PATH_PREFIXES = (
    ('percection_tool/weights', 'percection_tool/sam_road/weights'),
    ('percection_tool/vendor/sam_road', 'percection_tool/sam_road/vendor'),
    ('map_preprocess/weights', 'percection_tool/sam_road/weights'),
    ('map_preprocess/vendor/sam_road', 'percection_tool/sam_road/vendor'),
    ('map_preprocess/output', 'output/map_preprocess'),
    ('vlm/runtime', 'output/vlm/runtime'),
    ('vlm/log', 'output/vlm/log'),
    ('vlm/output', 'output/vlm/output'),
    ('vlm/requirements.txt', 'requirements.txt'),
    ('map_preprocess/run_agent.sh', 'agent/run_agent.sh'),
    ('map_preprocess', 'agent/map_preprocess'),
    ('vlm', 'agent/vlm'),
    ('kitti.png', 'data/input/kitti.png'),
    ('kitti_raw_0930_0027.tif', 'data/input/kitti_raw_0930_0027.tif'),
    ('baotou_rgb_0p2m.tif', 'data/input/baotou_rgb_0p2m.tif'),
    # 输入迁到 input_geo_img/ 前的路径（已核对 SHA256 一致），旧任务单独重跑 Node 5-9 时仍能找到原图
    ('data/Kitti遥感数据/kitti_raw_1003_0027.tif', 'input_geo_img/kitti_1003_0027.tif'),
    ('data/Kitti遥感数据/kitti_raw_1003_0034.tif', 'input_geo_img/kitti_1003_0034.tif'),
    ('data/河西遥感/hexi_dom_0.05m.tif', 'input_geo_img/hexi_dom.tif'),
    ('data/河西遥感/hexi_dsm.tif', 'input_geo_img/hexi_dsm.tif'),
)


def resolve_path(value: str | Path) -> Path:
    """相对路径固定按项目根解析；保留虚拟环境符号链接。"""
    path = Path(value).expanduser()
    absolute = Path(os.path.abspath(path if path.is_absolute() else PROJECT_ROOT / path))
    try:
        relative = absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return absolute
    for old, new in LEGACY_PATH_PREFIXES:
        if relative == old or relative.startswith(old + '/'):
            return PROJECT_ROOT / (new + relative[len(old):])
    return absolute


def relative_path(value: str | Path) -> str:
    return Path(os.path.relpath(resolve_path(value), PROJECT_ROOT)).as_posix()


def write_json(path: str | Path, value: Any) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fsync_path(path: str | Path) -> None:
    """把文件内容刷到磁盘；缓存“写临时文件再改名”前调用，避免断电后留下截断的文件。"""
    with resolve_path(path).open('rb') as handle:
        os.fsync(handle.fileno())


def read_json(path: str | Path) -> Any:
    return json.loads(resolve_path(path).read_text(encoding='utf-8'))
