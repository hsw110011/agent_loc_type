"""final_summary：区分有退化 / 无退化 / 不完整，没跑完或有失败时不能当作无退化。"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.map_preprocess.utils.final_summary import build_final_summary


def clean_state(**overrides) -> dict:
    state = {'block_features': {'status': 'completed', 'block_count': 10, 'height_source': 'dsm', 'degraded_blocks': []},
             'vlm_review': {'status': 'completed', 'reviewed_blocks': 0, 'degraded_scene_count': 0, 'errors': [],
                            'unverified_block_ids': [], 'rejected_block_ids': [], 'removed_block_ids': []},
             'vlm_expansion': {'status': 'completed', 'errors': []},
             'degradation_fusion': {'status': 'completed', 'region_count': 0, 'excluded_count': 0, 'rejected_count': 0},
             'localization_recommendation': {'status': 'completed', 'recommended_count': 0, 'failed_count': 0, 'method_counts': {}}}
    for key, value in overrides.items():
        state[key] = {**state[key], **value}
    return state


class FinalSummaryTests(unittest.TestCase):
    def test_all_clean_is_no_degradation(self):
        summary = build_final_summary(clean_state())
        self.assertEqual((summary['conclusion'], summary['complete'], summary['issues']), ('no_degradation', True, []))
        self.assertIn('未发现退化区域', summary['message'])

    def test_zero_blocks_is_incomplete(self):
        summary = build_final_summary({'block_features': {'status': 'completed', 'block_count': 0,
                                                         'review_candidate_block_ids': []}})
        self.assertEqual((summary['conclusion'], summary['complete']), ('incomplete', False))
        self.assertEqual(summary['issues'][0]['kind'], 'no_blocks')

    def test_one_confirmed_branch_does_not_hide_an_unresolved_branch(self):
        state = clean_state(block_features={'degraded_blocks': ['E1_B0']},
                            vlm_review={'degraded_scene_count': 1, 'unresolved_branch_block_ids': ['E1_B0']},
                            degradation_fusion={'region_count': 1})
        summary = build_final_summary(state)
        self.assertEqual((summary['conclusion'], summary['complete']), ('degraded', False))
        self.assertEqual(summary['issues'][0]['ids'], ['E1_B0'])

    def test_regions_are_degraded_with_methods(self):
        summary = build_final_summary(clean_state(degradation_fusion={'region_count': 2},
                                                  localization_recommendation={'recommended_count': 2, 'method_counts': {'gps': 1, 'wheel_imu_odometry': 1}}))
        self.assertEqual((summary['conclusion'], summary['complete']), ('degraded', True))
        self.assertEqual(summary['method_counts'], {'gps': 1, 'wheel_imu_odometry': 1})

    def test_degraded_with_failures_is_marked_incomplete(self):
        summary = build_final_summary(clean_state(degradation_fusion={'region_count': 1},
                                                  localization_recommendation={'status': 'partial', 'failed_count': 1, 'failed_region_ids': ['Merged_2']}))
        self.assertEqual((summary['conclusion'], summary['complete']), ('degraded', False))
        self.assertEqual(summary['issues'][0]['ids'], ['Merged_2'])

    def test_failures_without_regions_are_incomplete(self):
        candidate = {'degraded_blocks': ['E2_B0']}   # 有候选才会走到 Node 6
        confirmed = {'degraded_scene_count': 1}   # 有确认退化才会走到 Node 7-9
        cases = {'node6': clean_state(block_features=candidate, vlm_review={'errors': [{'block_id': 'E2_B0', 'review_type': 'height', 'error': 'x'}]}),
                 'node8': clean_state(block_features=candidate, vlm_review=confirmed, degradation_fusion={'excluded_count': 1})}
        for stage, state in cases.items():
            with self.subTest(stage):
                summary = build_final_summary(state)
                self.assertEqual(summary['conclusion'], 'incomplete')
                self.assertEqual(summary['issues'][0]['stage'], stage)
                self.assertIn('不能当作无退化', summary['message'])

    def test_node7_request_failure_is_not_an_issue(self):
        # Node 7 请求失败时该支路保留 Node 6 基准尺寸并进入融合，不算不完整
        state = clean_state(block_features={'degraded_blocks': ['E3_B0']}, vlm_review={'degraded_scene_count': 1},
                            vlm_expansion={'errors': [{'block_id': 'E3_B0', 'review_type': 'height', 'error': 'x'}]},
                            degradation_fusion={'region_count': 1})
        summary = build_final_summary(state)
        self.assertEqual((summary['conclusion'], summary['complete'], summary['issues']), ('degraded', True, []))

    def test_unverified_and_confirmed_without_regions(self):
        self.assertEqual(build_final_summary(clean_state(vlm_review={'unverified_block_ids': ['E4_B0']}))['issues'][0]['kind'], 'unverified')
        summary = build_final_summary(clean_state(vlm_review={'degraded_scene_count': 1}))
        self.assertEqual((summary['conclusion'], summary['issues']), ('incomplete', []))

    def test_stage_not_run_or_dry_run(self):
        state = clean_state(block_features={'degraded_blocks': ['E1_B0']}, vlm_review={'status': 'dry_run'})
        del state['localization_recommendation']
        issue = build_final_summary(state)['issues'][0]
        self.assertEqual((issue['kind'], issue['ids']), ('stage_not_run', ['node6', 'node9']))
        self.assertEqual(build_final_summary({})['conclusion'], 'incomplete')

    def test_early_exit_after_node5_or_node6(self):
        # Node 5 没有候选：Node 6-9 不需要运行，仍然是“无退化”
        state = clean_state()
        for key in ('vlm_review', 'vlm_expansion', 'degradation_fusion', 'localization_recommendation'):
            del state[key]
        summary = build_final_summary(state)
        self.assertEqual((summary['conclusion'], summary['early_exit']['stage']), ('no_degradation', 'node5'))
        self.assertIn('流程提前结束', summary['message'])
        # Node 6 复核后没有确认退化：Node 7-9 不需要运行；复核失败仍是不完整
        state = clean_state(block_features={'degraded_blocks': ['E1_B0']}, vlm_review={'reviewed_blocks': 1, 'rejected_block_ids': ['E1_B0']})
        for key in ('vlm_expansion', 'degradation_fusion', 'localization_recommendation'):
            del state[key]
        self.assertEqual((build_final_summary(state)['conclusion'], build_final_summary(state)['early_exit']['stage']), ('no_degradation', 'node6'))
        state['vlm_review']['unverified_block_ids'] = ['E1_B0']
        self.assertEqual(build_final_summary(state)['conclusion'], 'incomplete')
        # 有确认退化时不提前结束，缺 Node 7-9 就是不完整
        state = clean_state(block_features={'degraded_blocks': ['E1_B0']}, vlm_review={'degraded_scene_count': 1})
        del state['localization_recommendation']
        summary = build_final_summary(state)
        self.assertEqual((summary['conclusion'], summary['early_exit']), ('incomplete', None))


class GraphEarlyExitTests(unittest.TestCase):
    """Node 5 / Node 6 之后的条件边：没有可做的就直接汇总 manifest。"""

    def run_graph(self, candidates: list[str], confirmed: int) -> list[str]:
        from agent.map_preprocess.global_config import GlobalPreprocessConfig
        from agent.map_preprocess.graph import build_graph
        ran = []

        def stub(name: str, update: dict):
            def function(state, **kwargs):
                ran.append(name)
                return update
            function.__name__ = name
            return function
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        config = GlobalPreprocessConfig(segmenter='dummy', block_features_enabled=True, vlm_review_enabled=True, vlm_expansion_enabled=True,
                                        block_size_m=64, degradation_fusion_enabled=True, vlm_localization_enabled=True)
        stubs = {
            'agent.map_preprocess.graph.load_image_and_resolution': stub('load_image_and_resolution', {}),
            'agent.map_preprocess.graph.extract_road_graph': stub('extract_road_graph', {}),
            'agent.map_preprocess.graph.clean_road_graph': stub('clean_road_graph', {}),
            'agent.map_preprocess.graph.build_blocks_and_groups': stub('build_blocks_and_groups', {}),
            'agent.map_preprocess.graph.refresh_global_manifest': stub('refresh_global_manifest', {}),
            'agent.map_preprocess.nodes.block_features.compute_block_features': stub('compute_block_features', {'block_features': {'status': 'completed', 'degraded_blocks': candidates}}),
            'agent.map_preprocess.nodes.vlm_review.vlm_review_blocks': stub('vlm_review_blocks', {'vlm_review': {'status': 'completed', 'degraded_scene_count': confirmed}}),
            'agent.map_preprocess.nodes.vlm_expansion.vlm_expand_blocks': stub('vlm_expand_blocks', {}),
            'agent.map_preprocess.nodes.degradation_fusion.fuse_degraded_patches': stub('fuse_degraded_patches', {}),
            'agent.map_preprocess.nodes.localization_recommendation.recommend_localization': stub('recommend_localization', {}),
        }
        patches = [mock.patch(target, function) for target, function in stubs.items()]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        build_graph(config=config).invoke({'task_id': 't', 'output_dir': folder.name})
        return ran

    def test_no_candidates_skips_node6_to_node9(self):
        self.assertEqual(self.run_graph([], 0)[-2:], ['compute_block_features', 'refresh_global_manifest'])

    def test_no_confirmed_degradation_skips_node7_to_node9(self):
        self.assertEqual(self.run_graph(['E1_B0'], 0)[-3:], ['compute_block_features', 'vlm_review_blocks', 'refresh_global_manifest'])

    def test_confirmed_degradation_runs_everything(self):
        self.assertEqual(self.run_graph(['E1_B0'], 1)[-6:], ['compute_block_features', 'vlm_review_blocks', 'vlm_expand_blocks',
                                                              'fuse_degraded_patches', 'recommend_localization', 'refresh_global_manifest'])


if __name__ == '__main__':
    unittest.main()
