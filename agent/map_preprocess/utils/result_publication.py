"""同步已发布的阶段结果，并撤下依赖旧输入的下游产物引用。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .. import output_layout as layout
from ..io import read_json, write_json
from .final_summary import STAGES, build_final_summary


STAGE_FILES = {
    'block_features': ('compute_block_features', ('block_features', 'blocks_degraded', 'block_images', 'degraded_sheets')),
    'vlm_review': ('vlm_review_blocks', ('vlm_review', 'degraded_scenes', 'vlm_review_overview')),
    'vlm_expansion': ('vlm_expand_blocks', ('vlm_expansion', 'degradation_extents', 'vlm_expansion_overview')),
    'degradation_fusion': ('fuse_degraded_patches', ('degradation_regions', 'degradation_fusion_overview')),
    'localization_recommendation': ('recommend_localization', ('localization_recommendations', 'localization_recommendation_overview')),
}


@contextmanager
def protect_stage_outputs(manifest_path: Path, paths: tuple[Path, ...], *, keep: tuple[Path, ...] = ()):
    """暂存旧产物。正式重跑中途失败时恢复原文件与发布记录。"""
    state_path = layout.state_dir(layout.task_dir_of_manifest(manifest_path))/'state.json'
    with TemporaryDirectory(prefix='.rerun-', dir=manifest_path.parent) as directory:
        saved = []
        try:
            for path in (*paths, manifest_path, state_path):
                if not path.exists():
                    saved.append((path, None))
                    continue
                backup = Path(directory)/str(len(saved))
                if path in keep and path.is_dir():
                    shutil.copytree(path, backup)
                elif path in (*keep, manifest_path, state_path):
                    shutil.copy2(path, backup)
                else:
                    path.rename(backup)
                saved.append((path, backup))
            yield
        except BaseException:
            for path, backup in reversed(saved):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
                if backup is not None:
                    backup.rename(path)
            raise


def publish_stage(manifest_path: str | Path, manifest: dict, stage: str) -> None:
    stages = [key for key, _, _ in STAGES]
    affected = stages[stages.index(stage):]
    files, extensions = manifest['files'], manifest['extensions']
    for key in affected[1:]:
        extensions.pop(key, None)
        for filename in STAGE_FILES[key][1]:
            files.pop(filename, None)

    previous = read_json(manifest_path)
    finalized = 'final_summary' in previous or 'final_summary' in manifest
    if finalized:
        stale_snapshots = {STAGE_FILES[key][0] for key in affected} | {'refresh_global_manifest'}
        if snapshots := previous['files'].get('node_states'):
            files['node_states'] = {name: path for name, path in snapshots.items() if name[3:] not in stale_snapshots}
        manifest['final_summary'] = build_final_summary(extensions)

    write_json(manifest_path, manifest)
    if finalized:
        state_path = layout.state_dir(layout.task_dir_of_manifest(manifest_path))/'state.json'
        if state_path.is_file():
            state = read_json(state_path)
            for key in affected[1:]:
                state.pop(key, None)
            state[stage] = {key: value for key, value in extensions[stage].items() if key != 'removed_blocks'}
            if stage == 'block_features':
                state['image'] = manifest['image']
                state['dsm_path'] = (manifest['image'].get('dsm') or {}).get('path')
                state['config'] = manifest['config']
            state['final_summary'] = manifest['final_summary']
            write_json(state_path, state)
