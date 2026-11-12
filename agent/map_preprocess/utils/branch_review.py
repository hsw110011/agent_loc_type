"""可注入的分支复核协议；没有 VLM 客户端或模型调用。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, TypedDict
import warnings


class EdgeInfo(TypedDict):
    edge_id: str
    edge_score: float
    polyline_xy: list[list[float]]
    length_m: float


@dataclass(frozen=True)
class EdgeDecision:
    edge_id: str
    is_main_road: bool
    road_type: Literal['main', 'secondary', 'local', 'non_road']
    keep: bool
    reason: str


class BranchReviewer(Protocol):
    def review(self, image_path: str, annotated_image_path: str,
               edges: list[EdgeInfo]) -> list[EdgeDecision]: ...


class PassThroughReviewer:
    def review(self, image_path: str, annotated_image_path: str,
               edges: list[EdgeInfo]) -> list[EdgeDecision]:
        return [EdgeDecision(e['edge_id'], True, 'main', True, 'pass_through') for e in edges]


class RuleBasedReviewer:
    def __init__(self, top_n: int):
        self.top_n = top_n

    def review(self, image_path: str, annotated_image_path: str,
               edges: list[EdgeInfo]) -> list[EdgeDecision]:
        keep = {e['edge_id'] for e in sorted(edges, key=lambda e: (-e['edge_score'], e['edge_id']))[:self.top_n]}
        return [EdgeDecision(e['edge_id'], e['edge_id'] in keep,
                             'main' if e['edge_id'] in keep else 'local', e['edge_id'] in keep,
                             'rule_top_n') for e in edges]


def validated_decisions(edges: list[dict], decisions: list) -> list[dict]:
    by_id = {}
    for decision in decisions if isinstance(decisions, list) else []:
        value = asdict(decision) if isinstance(decision, EdgeDecision) else decision
        if isinstance(value, dict) and isinstance(value.get('edge_id'), str):
            by_id.setdefault(value['edge_id'], []).append(value)
    result = []
    for edge in edges:
        candidates = by_id.get(edge['edge_id'], [])
        valid = len(candidates) == 1
        value = candidates[0] if valid else {}
        valid = (valid and type(value.get('keep')) is bool and type(value.get('is_main_road')) is bool
                 and isinstance(value.get('road_type'), str)
                 and value['road_type'] in {'main', 'secondary', 'local', 'non_road'}
                 and isinstance(value.get('reason'), str))
        if not valid:
            warnings.warn(f"分支 {edge['edge_id']} 复核决策缺失或非法，默认保留", RuntimeWarning)
            value = asdict(EdgeDecision(edge['edge_id'], True, 'main', True, 'invalid_decision_keep'))
        result.append(value)
    return result
