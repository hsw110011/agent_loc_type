"""整次运行的最终结论（final_summary）：只汇总各节点已有的计数，不改变任何判断。

conclusion 三选一：
- degraded：Node 8 产出了融合区域（附区域数、Node 9 推荐方式统计）；此时仍可能有 issues，complete=False 表示还有块没判断成功；
- no_degradation：Node 5-9 全部跑完（或因没有候选 / 没有确认退化而提前结束），没有任何失败或未确认的块，且没有确认的退化；
- incomplete：没有融合区域，但有节点没跑、VLM 判断失败或请求出错——不能当作“无退化”。
"""
from __future__ import annotations

from typing import Any

# 得出确定结论所需的节点：state 键、节点编号、中文名
STAGES = (('block_features', 'node5', '块级指标'), ('vlm_review', 'node6', 'VLM 退化复核'),
          ('vlm_expansion', 'node7', '退化尺寸扩展'), ('degradation_fusion', 'node8', '退化区域融合'),
          ('localization_recommendation', 'node9', '定位方式推荐'))
DONE = ('completed', 'partial')


def _issue(stage: str, kind: str, label: str, ids: list[str] | None = None, count: int | None = None) -> dict:
    ids = sorted(set(ids or []))
    return {'stage': stage, 'kind': kind, 'label': label, 'count': len(ids) if count is None else count, 'ids': ids}


def early_exit(state: dict[str, Any]) -> dict | None:
    """没有后续可做时流程提前结束：Node 5 没有可送审的块，或 Node 6 没有确认任何退化。

    返回 {'after': 结束前最后一个节点, 'stage', 'reason'}；没有提前结束条件时为 None。
    """
    features, review = state.get('block_features') or {}, state.get('vlm_review') or {}
    candidates = features.get('degraded_blocks', features.get('review_candidate_block_ids'))
    if features.get('status') == 'completed' and not candidates:
        return {'after': 'compute_block_features', 'stage': 'node5', 'reason': 'no_candidates', 'label': 'Node 5 没有可送审的道路块'}
    if review.get('status') in DONE and not review.get('degraded_scene_count'):
        return {'after': 'vlm_review_blocks', 'stage': 'node6', 'reason': 'no_confirmed_degradation', 'label': 'Node 6 没有确认的退化'}
    return None


def build_final_summary(state: dict[str, Any]) -> dict:
    issues: list[dict] = []
    missing = []
    exit_info = early_exit(state)
    # 提前结束后，之后的节点本来就不需要运行，不算缺失
    required = STAGES if exit_info is None else STAGES[:[s for _, s, _ in STAGES].index(exit_info['stage'])+1]
    for key, stage, name in required:
        status = (state.get(key) or {}).get('status')
        if status not in DONE:
            missing.append({'stage': stage, 'name': name, 'status': status or 'not_run', 'reason': (state.get(key) or {}).get('reason')})
    if missing:
        issues.append({'stage': 'pipeline', 'kind': 'stage_not_run', 'label': '节点未运行、被跳过或为 dry_run',
                       'count': len(missing), 'ids': [item['stage'] for item in missing], 'stages': missing})

    features = state.get('block_features') or {}
    if features.get('status') in DONE:
        if features.get('block_count') == 0:
            issues.append(_issue('node5', 'no_blocks', '未生成可分析的道路块', count=1))
        if unavailable := features.get('unavailable_review_data'):
            issues.append({**_issue('node5', 'no_data', '部分道路块缺少送审数据', list(unavailable)),
                           'reasons_by_block': unavailable})
    review = state.get('vlm_review') or {}
    if review.get('status') in DONE:
        if review.get('errors'):
            issues.append(_issue('node6', 'request_failed', 'VLM 复核请求失败', [e['block_id'] for e in review['errors']]))
        unverified = set(review.get('unverified_block_ids', ())) | set(review.get('unresolved_branch_block_ids', ()))
        if unverified:
            issues.append(_issue('node6', 'unverified', 'VLM 无法确认部分块或支路是否退化', list(unverified)))

    # Node 7 请求失败不算不完整：该支路保留 Node 6 基准尺寸并进入融合，失败明细仍在 state/vlm_expansion.json 的 errors。

    fusion = state.get('degradation_fusion') or {}
    regions = fusion.get('region_count', 0) if fusion.get('status') in DONE else 0
    if fusion.get('status') in DONE:
        if fusion.get('rejected_count'):
            issues.append(_issue('node8', 'invalid_patch', '无效 Patch 被丢弃', count=fusion['rejected_count']))
        if fusion.get('excluded_count'):
            issues.append(_issue('node8', 'not_fused', '已确认退化但未进入融合（最大确认档类型不在融合范围或没有确认尺寸）',
                                 [i.rsplit('_', 1)[0] for i in fusion.get('excluded_patch_ids', [])], count=fusion['excluded_count']))

    recommendation = state.get('localization_recommendation') or {}
    if recommendation.get('status') in DONE and recommendation.get('failed_count'):
        issues.append(_issue('node9', 'request_failed', '定位推荐失败', recommendation.get('failed_region_ids')))

    confirmed = review.get('degraded_scene_count', 0) if review.get('status') in DONE else 0
    if regions:
        conclusion = 'degraded'
    elif issues or confirmed:  # 已确认退化却没有融合区域，同样不能算无退化
        conclusion = 'incomplete'
    else:
        conclusion = 'no_degradation'

    candidates = features.get('degraded_blocks', features.get('review_candidate_block_ids'))
    counts = {'blocks': features.get('block_count'), 'candidate_blocks': len(candidates or []),
              'reviewed_blocks': review.get('reviewed_blocks'), 'confirmed_degraded_blocks': confirmed,
              'rejected_blocks': len(review.get('rejected_block_ids') or []),
              'removed_undrivable_blocks': len(review.get('removed_block_ids') or []),
              'degradation_regions': regions, 'recommended_regions': recommendation.get('recommended_count', 0),
              'review_required_regions': recommendation.get('review_required_count', 0)}
    detail = '；'.join(f"{item['stage']} {item['label']} {item['count']}" for item in issues)
    if conclusion == 'degraded':
        methods = recommendation.get('method_counts') or {}
        message = f"发现 {regions} 个退化区域，推荐方式 {methods or '未生成'}"
        message += f"（结果不完整：{detail}）" if issues else ''
    elif conclusion == 'incomplete':
        message = f"结果不完整，不能当作无退化：{detail or f'已确认退化 {confirmed} 块但没有融合区域'}"
    else:
        message = f"未发现退化区域（共 {counts['blocks']} 块，候选 {counts['candidate_blocks']} 块，全部判断成功）"
    if exit_info is not None:
        message += f"；{exit_info['label']}，流程提前结束"
    return {'conclusion': conclusion, 'complete': not issues, 'message': message, 'counts': counts, 'early_exit': exit_info,
            'method_counts': recommendation.get('method_counts') or {}, 'issues': issues}
