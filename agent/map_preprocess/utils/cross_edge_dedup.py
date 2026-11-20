"""沿拓扑的初始块去重、精确跨边覆盖及可选的全局贪心布块。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
import warnings

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, box

from ..global_config import GlobalPreprocessConfig
from .global_blocks import make_block

# 仅用于浮点边界判断，不是物理缓冲区或可调道路阈值。
_EPS = 1e-9
_MIN_DUPLICATE_AREA_RATIO = .75


def arc_positions(length_m: float, spacing_m: float) -> np.ndarray:
    """固定间距，始终包含起终点，不重复尾点。"""
    positions = np.arange(math.floor(length_m/spacing_m)+1, dtype=float)*spacing_m
    if length_m-positions[-1] > _EPS:
        positions = np.r_[positions, length_m]
    else:
        positions[-1] = length_m
    return positions


def interpolate_positions(edge: dict, arcs: np.ndarray, resolution: float) -> np.ndarray:
    points = np.asarray(edge['polyline_xy'], dtype=float)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.column_stack([np.interp(arcs/resolution, lengths, points[:, axis]) for axis in (0, 1)])


class RoadPointIndex:
    """局部网格查询，避免分配 候选块数×全图道路点数 的稠密矩阵。"""
    def __init__(self, points: np.ndarray, size_px: float):
        self.points, self.size_px = points, size_px
        cells = defaultdict(list)
        for index, point in enumerate(points):
            cells[tuple(np.floor(point/size_px).astype(int))].append(index)
        self.cells = {cell: np.asarray(indices, dtype=np.int64) for cell, indices in cells.items()}

    def query(self, bbox: list[float]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        a, b = np.floor((np.asarray([x0, y0])-_EPS)/self.size_px).astype(int)
        c, d = np.floor((np.asarray([x1, y1])+_EPS)/self.size_px).astype(int)
        buckets = [self.cells[(x,y)] for x in range(a,c+1) for y in range(b,d+1) if (x,y) in self.cells]
        if not buckets:
            return np.empty(0, dtype=np.int64)
        indices = np.concatenate(buckets)
        points = self.points[indices]
        return indices[(points[:,0] >= x0-_EPS) & (points[:,0] <= x1+_EPS) &
                       (points[:,1] >= y0-_EPS) & (points[:,1] <= y1+_EPS)]


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]+_EPS:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def covered_arc_intervals(edge: dict, bbox: list[float], resolution: float) -> list[tuple[float, float]]:
    """逐原折线段做 slab 裁切，保留自交/回返的不同弧长位置及切点。"""
    intervals, offset = [], 0.
    for a, b in zip(edge['polyline_xy'], edge['polyline_xy'][1:]):
        delta = np.asarray(b, dtype=float)-a
        length = float(np.linalg.norm(delta))*resolution
        low, high = 0., 1.
        for axis in (0, 1):
            if delta[axis] == 0:
                if a[axis] < bbox[axis]-_EPS or a[axis] > bbox[axis+2]+_EPS:
                    high = -1.
                    break
            else:
                first = (bbox[axis]-a[axis])/delta[axis]
                last = (bbox[axis+2]-a[axis])/delta[axis]
                low, high = max(low, min(first,last)), min(high, max(first,last))
        if low <= high and length > 0:
            intervals.append((offset+low*length, offset+high*length))
        offset += length
    return merge_intervals(intervals)


@dataclass
class Candidate:
    edge_id: str
    arc_m: float
    x: float
    y: float
    covered: np.ndarray


def connect_covered_blocks(edges: list[dict], blocks: list[dict]) -> None:
    """按覆盖区间串联，并只在模型共享节点处跨边连接。"""
    by_id = {block['block_id']: block for block in blocks}
    loops = {edge['edge_id'] for edge in edges if edge['u'] == edge['v']}
    segments = defaultdict(list)
    predecessors = {identifier: set() for identifier in by_id}
    successors = {identifier: set() for identifier in by_id}
    for block in blocks:
        for edge_id, start, end in block['covered_segments']:
            segments[edge_id].append((start, end, block['block_id']))
    endpoints = defaultdict(list)
    for edge in edges:
        ordered = sorted(segments[edge['edge_id']])
        edge['block_ids'] = [b['block_id'] for b in blocks if b['edge_id'] == edge['edge_id']]
        # 每条边的遍历列表可以引用中心归属于邻边的块；同一块每边只出现一次。
        edge['covering_block_ids'] = list(dict.fromkeys(identifier for _, _, identifier in ordered))
        for (_, _, previous), (_, _, following) in zip(ordered, ordered[1:]):
            if previous != following:
                successors[previous].add(following)
                predecessors[following].add(previous)
        for start, end, identifier in ordered:
            if start <= _EPS:
                endpoints[edge['u']].append((edge['edge_id'], identifier, 'predecessor'))
            if end >= edge['length_m']-_EPS:
                endpoints[edge['v']].append((edge['edge_id'], identifier, 'successor'))
    for entries in endpoints.values():
        for edge_id, identifier, field in entries:
            targets = predecessors if field == 'predecessor' else successors
            targets[identifier].update(other for other_edge, other, _ in entries
                                      if other != identifier and (other_edge != edge_id or edge_id in loops))
    for identifier, block in by_id.items():
        block['predecessor_ids'] = sorted(predecessors[identifier])
        block['successor_ids'] = sorted(successors[identifier])


def _greedy_select(candidates: list[Candidate], count: int, config: GlobalPreprocessConfig) -> tuple[list[Candidate], np.ndarray]:
    """全局贪心：每轮选新增覆盖最多的候选，直到覆盖率达到 min_road_coverage。"""
    heap = [(-len(c.covered), c.x, c.y, c.edge_id, c.arc_m, i) for i, c in enumerate(candidates)]
    heapq.heapify(heap)
    cover_counts, selected = np.zeros(count, dtype=np.int64), []
    covered_count = 0
    while heap and covered_count < config.min_road_coverage*count:
        previous_gain, x, y, edge_id, arc, candidate_index = heapq.heappop(heap)
        candidate = candidates[candidate_index]
        gain = int(np.count_nonzero(cover_counts[candidate.covered] == 0))
        # 未更新增益是上界；惰性重算保持与每轮全量贪心相同的选择和平局规则。
        if gain != -previous_gain:
            heapq.heappush(heap, (-gain,x,y,edge_id,arc,candidate_index))
            continue
        if gain == 0:
            break
        selected.append(candidate)
        covered_count += gain
        cover_counts[candidate.covered] += 1
    return selected, cover_counts


def _set_covered_segments(blocks: list[dict], edges: list[dict], resolution: float) -> None:
    tree = STRtree([LineString(edge['polyline_xy']) for edge in edges])
    for block in blocks:
        bbox = block['global_bbox']
        x0, y0, x1, y1 = bbox
        neighbors = sorted(tree.query(box(x0-_EPS, y0-_EPS, x1+_EPS, y1+_EPS)))
        block['covered_segments'] = [(edges[i]['edge_id'], start, end) for i in neighbors
            for start, end in covered_arc_intervals(edges[i], bbox, resolution)]


def _prune_overlapping_blocks(blocks: list[dict], edges: list[dict]) -> list[dict]:
    """仅删除高重叠邻框能接手全部精确道路区间的块。"""
    boxes = [box(*block['global_bbox']) for block in blocks]
    tree = STRtree(boxes)
    edge_lengths = {edge['edge_id']: edge['length_m'] for edge in edges}
    # 同等覆盖时优先保留长边和靠前的块；逐个删除以免两块相互替代后同时消失。
    order = sorted(range(len(blocks)), reverse=True, key=lambda i: (
        -edge_lengths[blocks[i]['edge_id']], blocks[i]['edge_id'], blocks[i]['arc_length_m']))
    removed = set()
    for i in order:
        bbox = boxes[i]
        neighbors = [j for j in tree.query(bbox) if j != i and j not in removed
                     and bbox.intersection(boxes[j]).area >= bbox.area*_MIN_DUPLICATE_AREA_RATIO]
        if not neighbors:
            continue
        intervals = defaultdict(list)
        for j in neighbors:
            for edge_id, start, end in blocks[j]['covered_segments']:
                intervals[edge_id].append((start, end))
        shared = {edge_id: merge_intervals(parts) for edge_id, parts in intervals.items()}
        if all(any(a-_EPS <= start and end <= b+_EPS for a, b in shared.get(edge_id, ()))
               for edge_id, start, end in blocks[i]['covered_segments']):
            removed.add(i)
    return [block for i, block in enumerate(blocks) if i not in removed]


def prune_redundant(selected: list[Candidate], cover_counts: np.ndarray, config: GlobalPreprocessConfig,
                    side: float) -> tuple[list[Candidate], int]:
    """按独占采样点剪除冗余框，同时要求保留邻框至少重叠 75% 面积。"""
    if not config.dedup_prune_enabled or not selected:
        return selected, 0
    boxes = [box(c.x-side/2, c.y-side/2, c.x+side/2, c.y+side/2) for c in selected]
    tree = STRtree(boxes)
    neighbors = [{j for j in tree.query(bbox) if j != i
                  and bbox.intersection(boxes[j]).area >= bbox.area*_MIN_DUPLICATE_AREA_RATIO}
                 for i, bbox in enumerate(boxes)]
    kept = set(range(len(selected)))
    total = len(cover_counts)
    covered = int(np.count_nonzero(cover_counts))
    while eligible := [i for i in kept if neighbors[i] & kept]:
        unique = {i: int(np.count_nonzero(cover_counts[selected[i].covered] == 1)) for i in eligible}
        index = min(eligible, key=lambda i: (unique[i], len(selected[i].covered),
                                            selected[i].edge_id, selected[i].arc_m, i))
        loss = unique[index]
        if (loss*config.road_sample_spacing_m >= config.dedup_prune_max_loss_m
                or covered-loss < config.min_road_coverage*total-_EPS):
            break
        cover_counts[selected[index].covered] -= 1
        covered -= loss
        kept.remove(index)
    return [selected[i] for i in sorted(kept)], len(selected)-len(kept)


def cross_edge_dedup(edges: list[dict], original_blocks: list[dict], image: dict, tiles: list[dict],
                     resolution: float, config: GlobalPreprocessConfig) -> tuple[list[dict], list[dict], dict]:
    """返回保留块、逐边覆盖指标和统计；不保存未选候选或道路采样点。"""
    ordered_edges = sorted(edges, key=lambda edge: edge['edge_id'])
    samples, sample_ranges = [], {}
    count = 0
    for edge in ordered_edges:
        points = interpolate_positions(edge, arc_positions(edge['length_m'], config.road_sample_spacing_m), resolution)
        samples.append(points)
        sample_ranges[edge['edge_id']] = (count, count+len(points))
        count += len(points)
    points = np.concatenate(samples) if samples else np.empty((0,2))
    side = config.block_size_m/resolution
    index = RoadPointIndex(points, side)
    before_counts = np.zeros(count, dtype=np.int64)
    for block in original_blocks:
        before_counts[index.query(block['global_bbox'])] += 1
    if config.dedup_method == 'walk':
        blocks = [dict(block) for block in sorted(original_blocks, key=lambda b: (b['edge_id'], b['arc_length_m']))]
        _set_covered_segments(blocks, ordered_edges, resolution)
        candidate_count = len(blocks)
        if config.dedup_prune_enabled:
            blocks = _prune_overlapping_blocks(blocks, ordered_edges)
        pruned = candidate_count-len(blocks)
        sequences = defaultdict(int)
        cover_counts = np.zeros(count, dtype=np.int64)
        for block in blocks:
            edge_id = block['edge_id']
            sequence = sequences[edge_id]
            block.update(seq=sequence, block_id=f'E{edge_id}_B{sequence}')
            sequences[edge_id] += 1
            cover_counts[index.query(block['global_bbox'])] += 1
    else:
        candidates = []
        for edge in ordered_edges:
            arcs = arc_positions(edge['length_m'], config.dedup_candidate_spacing_m)
            for arc, (x,y) in zip(arcs, interpolate_positions(edge, arcs, resolution)):
                candidates.append(Candidate(edge['edge_id'], float(arc), float(x), float(y),
                                            index.query([x-side/2,y-side/2,x+side/2,y+side/2])))
        selected, cover_counts = _greedy_select(candidates, count, config)
        candidate_count = len(candidates)
        selected, pruned = prune_redundant(selected, cover_counts, config, side)
        blocks = []
        for edge in ordered_edges:
            owned = sorted((c for c in selected if c.edge_id == edge['edge_id']), key=lambda c: (c.arc_m,c.x,c.y))
            blocks.extend(make_block(edge, candidate.arc_m, sequence, len(owned), image, tiles, resolution, config)
                          for sequence, candidate in enumerate(owned))
        _set_covered_segments(blocks, ordered_edges, resolution)
    covered_count = int(np.count_nonzero(cover_counts))
    road_coverage = covered_count/count if count else 1.
    if config.dedup_method == 'greedy' and road_coverage+_EPS < config.min_road_coverage:
        warnings.warn(f'cross_edge_dedup 无新增覆盖，覆盖率 {road_coverage:.6f} 低于 min_road_coverage', RuntimeWarning)
    connect_covered_blocks(edges, blocks)
    coverage = []
    for edge in ordered_edges:
        intervals = [(start,end) for block in blocks for identifier,start,end in block['covered_segments']
                     if identifier == edge['edge_id']]
        union = merge_intervals(intervals)
        start, stop = sample_ranges[edge['edge_id']]
        coverage.append({'edge_id': edge['edge_id'],
            'coverage_ratio': min(1., sum(b-a for a,b in union)/edge['length_m']),
            'sample_coverage_ratio': float(np.count_nonzero(cover_counts[start:stop]))/(stop-start),
            'covered_intervals_m': union, 'block_count': len(edge['covering_block_ids']),
            'owned_block_count': len(edge['block_ids']), 'min_overlap_ratio': None, 'max_overlap_ratio': None})
    stats = {'enabled': True, 'method': config.dedup_method, 'blocks_before': len(original_blocks),
        'candidate_blocks': candidate_count, 'min_gain_m': None,
        'skipped_road_sample_count': 0, 'pruned_redundant_blocks': pruned,
        'blocks_after': len(blocks), 'road_sample_count': count, 'covered_road_sample_count': covered_count,
        'road_coverage': road_coverage, 'min_road_coverage': config.min_road_coverage,
        'mean_coverage_count_before': float(before_counts.mean()) if count else 0.,
        'mean_coverage_count': float(cover_counts.mean()) if count else 0.}
    return blocks, coverage, stats
