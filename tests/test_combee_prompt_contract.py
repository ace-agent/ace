"""Offline native-role contracts; no client, model download, or benchmark run.

Run from the repository root: python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from unittest.mock import patch

from ace.core.curator import Curator
from ace.core.reflector import Reflector
from ace.prompts.curator import (
    CURATOR_PROMPT, CURATOR_PROMPT_NO_GT,
    CURATOR_OPERATIONS_AGGREGATION_PROMPT,
)
from ace.prompts.reflector import REFLECTOR_PROMPT, REFLECTOR_PROMPT_NO_GT
from playbook_utils import apply_curator_operations


class CombeePromptContractTest(unittest.TestCase):
    def setUp(self):
        network = patch('socket.socket.connect', side_effect=AssertionError('network forbidden'))
        network.start()
        self.addCleanup(network.stop)
        self.logs = tempfile.TemporaryDirectory()
        self.addCleanup(self.logs.cleanup)
        # No SDK client is constructed. Only the model call boundary is mocked.
        self.curator = Curator(None, 'openai', 'offline-test')
        self.reflector = Reflector(None, 'openai', 'offline-test')
        self.playbook = '## STRATEGIES & INSIGHTS\n'
        self.add = {'type': 'ADD', 'section': 'STRATEGIES & INSIGHTS',
                    'content': 'Inspect stderr before retrying.'}
        self.unsupported = [{'type': kind, 'content': 'must not be applied'}
                            for kind in ('UPDATE', 'REMOVE', 'DELETE', 'MERGE')]
        self.info = {'synthetic': True}
        self.fields = dict(current_playbook=self.playbook,
                           recent_reflection='reflection-marker',
                           question_context='context-marker', current_step=7,
                           total_samples=60, token_budget=2048,
                           playbook_stats={'total_bullets': 0})

    def response(self, operations):
        return json.dumps({'reasoning': 'synthetic', 'operations': operations})

    def test_reflector_gt_no_gt_and_missing_gt_fallback(self):
        tags = [{'bullet_id': 'str-00001', 'tag': 'helpful'}]
        response = json.dumps({'bullet_tags': tags})
        for use_gt, truth in ((True, 'truth-marker'), (False, 'truth-marker'),
                              (True, None), (True, '')):
            with self.subTest(use_gt=use_gt, truth=truth):
                with patch('ace.core.reflector.timed_llm_call',
                           return_value=(response, self.info)) as call:
                    result = self.reflector.reflect(
                        question='task-marker', reasoning_trace='$ cat file\nstderr-marker',
                        predicted_answer='attempt-marker', ground_truth=truth,
                        environment_feedback='exit=1; test failed', bullets_used='str-00001',
                        use_ground_truth=use_gt, use_json_mode=True,
                        call_id='reflect-contract', log_dir=self.logs.name)
                values = ['task-marker', '$ cat file\nstderr-marker', 'attempt-marker']
                if use_gt and truth:
                    expected = REFLECTOR_PROMPT.format(
                        *values, truth, 'exit=1; test failed', 'str-00001')
                else:
                    expected = REFLECTOR_PROMPT_NO_GT.format(
                        *values, 'exit=1; test failed', 'str-00001')
                    self.assertNotIn('truth-marker', call.call_args.args[3])
                self.assertEqual(call.call_args.args[3], expected)
                self.assertEqual(call.call_args.kwargs['call_id'], 'reflect-contract')
                call.assert_called_once()
                self.assertEqual(result, (response, tags, self.info))

    def test_curate_and_proposal_gt_no_gt_templates(self):
        for method in ('curate', 'propose_operations'):
            for use_gt in (True, False):
                with self.subTest(method=method, use_gt=use_gt):
                    with patch('ace.core.curator.timed_llm_call',
                               return_value=(self.response([self.add]), self.info)) as call:
                        result = getattr(self.curator, method)(
                            **self.fields, use_ground_truth=use_gt, use_json_mode=True,
                            call_id='curate-contract', log_dir=self.logs.name)
                    template = CURATOR_PROMPT if use_gt else CURATOR_PROMPT_NO_GT
                    fields = dict(self.fields, playbook_stats=json.dumps(
                        self.fields['playbook_stats'], indent=2))
                    self.assertEqual(call.call_args.args[3], template.format(**fields))
                    self.assertEqual(call.call_args.kwargs['call_id'], 'curate-contract')
                    self.assertTrue(call.call_args.kwargs['use_json_mode'])
                    call.assert_called_once()
                    if method == 'curate':
                        self.assertIn(self.add['content'], result[0])
                        self.assertEqual(result[1:], (2, [self.add], self.info))
                    else:
                        self.assertEqual(result, ([self.add], self.info))

    def test_proposal_filters_unsupported_operations_without_applying(self):
        with patch('ace.core.curator.timed_llm_call', return_value=(
                self.response([self.add] + self.unsupported), self.info)):
            result = self.curator.propose_operations(**self.fields)
        self.assertEqual(result, ([self.add], self.info))
        self.assertEqual(self.fields['current_playbook'], self.playbook)

    def test_reducer_serializes_groups_filters_and_applies_once(self):
        groups = [{'group_id': 1, 'sample_numbers': [1], 'operations': [self.add]},
                  {'group_id': 2, 'sample_numbers': [2],
                   'operations': [dict(self.add, content='Inspect café logs.')]}]
        with patch('ace.core.curator.timed_llm_call', return_value=(
                self.response([self.add] + self.unsupported), self.info)) as call:
            operations, info = self.curator.aggregate_operations(
                current_playbook=self.playbook, operation_groups=groups,
                current_step=7, total_samples=60, token_budget=2048,
                playbook_stats={'total_bullets': 0}, use_json_mode=True,
                call_id='reduce-contract', log_dir=self.logs.name)
        expected = CURATOR_OPERATIONS_AGGREGATION_PROMPT.format(
            current_playbook=self.playbook,
            proposed_operations=json.dumps(groups, indent=2, ensure_ascii=False),
            current_step=7, total_samples=60, token_budget=2048,
            playbook_stats=json.dumps({'total_bullets': 0}, indent=2))
        self.assertEqual(call.call_args.args[3], expected)
        self.assertEqual(call.call_args.kwargs['call_id'], 'reduce-contract')
        call.assert_called_once()
        self.assertEqual((operations, info), ([self.add], self.info))
        updated, next_id = apply_curator_operations(self.playbook, operations, 1)
        self.assertEqual(updated.count(self.add['content']), 1)
        self.assertNotIn('must not be applied', updated)
        self.assertEqual(next_id, 2)

    def test_invalid_model_outputs_are_not_proposed_or_reduced(self):
        for response in ('not JSON', self.response([{'type': 'ADD'}]),
                         json.dumps({'reasoning': 'synthetic', 'operations': 'invalid'})):
            for method in ('propose_operations', 'aggregate_operations'):
                with self.subTest(response=response, method=method):
                    args = dict(self.fields)
                    if method == 'aggregate_operations':
                        args.pop('recent_reflection')
                        args.pop('question_context')
                        args['operation_groups'] = []
                    with patch('ace.core.curator.timed_llm_call',
                               return_value=(response, self.info)):
                        result = getattr(self.curator, method)(**args, log_dir=self.logs.name)
                    self.assertEqual(result, ([], self.info))

    def test_vanilla_curate_reports_but_does_not_apply_unsupported_operations(self):
        # Unlike proposal/reducer, curate returns the validated raw operations.
        operations = [self.add] + self.unsupported
        with patch('ace.core.curator.timed_llm_call', return_value=(
                self.response(operations), self.info)):
            playbook, next_id, reported, info = self.curator.curate(
                **self.fields, log_dir=self.logs.name)
        self.assertEqual(reported, operations)
        self.assertEqual(info, self.info)
        self.assertIn(self.add['content'], playbook)
        self.assertNotIn('must not be applied', playbook)
        self.assertEqual(next_id, 2)

    def test_invalid_vanilla_curate_preserves_playbook_and_next_id(self):
        with patch('ace.core.curator.timed_llm_call', return_value=(
                self.response([{'type': 'ADD'}]), self.info)):
            result = self.curator.curate(**self.fields, next_global_id=23,
                                         log_dir=self.logs.name)
        self.assertEqual(result, (self.playbook, 23, [], self.info))

    def test_unsupported_only_reducer_does_not_allocate_ids(self):
        with patch('ace.core.curator.timed_llm_call', return_value=(
                self.response(self.unsupported), self.info)):
            operations, info = self.curator.aggregate_operations(
                current_playbook=self.playbook, operation_groups=[], current_step=7,
                total_samples=60, token_budget=2048, playbook_stats={},
                log_dir=self.logs.name)
        self.assertEqual((operations, info), ([], self.info))
        updated, next_id = apply_curator_operations(self.playbook, operations, 23)
        self.assertEqual(updated, self.playbook.strip())
        self.assertEqual(next_id, 23)


if __name__ == '__main__':
    unittest.main()
