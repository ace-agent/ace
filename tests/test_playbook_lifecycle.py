"""Synthetic lifecycle transactions; no model/client or network calls."""
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

from playbook_lifecycle import (
    prepare_playbook_candidate, publish_playbook_candidate,
    rollback_playbook_candidate, apply_playbook_lifecycle,
)
from playbook_utils import parse_playbook_line
from utils import count_tokens


BASE = (
    '## STRATEGIES & INSIGHTS\n'
    '[str-00001] helpful=4 harmful=1 :: Inspect stderr.\n'
    '[str-00002] helpful=2 harmful=3 :: Inspect the error log.\n'
    '[str-00003] helpful=0 harmful=1 :: Retry blindly.\n'
    '\n## OTHERS\n'
    '[misc-00004] helpful=1 harmful=0 :: Keep unrelated advice.\n'
)
UPDATE = {'type': 'UPDATE', 'bullet_id': 'str-00001', 'content': 'Inspect stderr before retrying.'}
MERGE = {'type': 'MERGE', 'source_ids': ['str-00001', 'str-00002'],
         'content': 'Inspect stderr and the error log before retrying.'}
DELETE = {'type': 'DELETE', 'bullet_id': 'str-00003'}
ADD = {'type': 'ADD', 'section': 'STRATEGIES & INSIGHTS', 'content': 'Record the exit code.'}


def bullets(text):
    return {b['id']: b for line in text.splitlines() if (b := parse_playbook_line(line))}


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        network = patch('socket.socket.connect', side_effect=AssertionError('network forbidden'))
        network.start()
        self.addCleanup(network.stop)

    def test_update_preserves_identity_counters_and_unrelated_text(self):
        candidate = prepare_playbook_candidate(BASE, [UPDATE], 8)
        updated, next_id, decision = publish_playbook_candidate(candidate, BASE, 8, 0)
        self.assertEqual(updated, BASE.replace(':: Inspect stderr.', ':: Inspect stderr before retrying.'))
        self.assertEqual(next_id, 8)
        self.assertEqual(bullets(updated)['str-00001']['helpful'], 4)
        self.assertEqual(decision['status'], 'accepted')

    def test_merge_retains_first_source_identity_sums_counts_and_delete_does_not_reuse_id(self):
        result, next_id, decision = apply_playbook_lifecycle(BASE, [MERGE, DELETE, ADD], 8, 999)
        result_bullets = bullets(result)
        self.assertEqual(set(result_bullets), {'str-00001', 'misc-00004', 'sai-00008'})
        self.assertEqual(result_bullets['str-00001']['helpful'], 6)
        self.assertEqual(result_bullets['str-00001']['harmful'], 4)
        self.assertEqual(result_bullets['str-00001']['content'], MERGE['content'])
        self.assertEqual(next_id, 9)
        self.assertIn('\n## OTHERS\n[misc-00004] helpful=1 harmful=0 :: Keep unrelated advice.\n', result)
        self.assertEqual(decision['status'], 'accepted')

    def test_invalid_batch_never_partially_applies_or_consumes_id(self):
        invalid = [
            {'type': 'CREATE_META'},
            {'type': 'REMOVE', 'bullet_id': 'str-00003'},
            dict(UPDATE, bullet_id='str-99999'),
            dict(UPDATE, bullet_id=None),
            dict(UPDATE, content='a\n[evil-00001] helpful=0 harmful=0 :: injected'),
            dict(UPDATE, content=''),
            dict(UPDATE, content=None),
            dict(ADD, section='missing'),
            dict(MERGE, source_ids=['str-00001', 'str-00001']),
            dict(MERGE, source_ids=['str-00001', 'misc-00004']),
            dict(MERGE, source_ids=['str-00001']),
            None,
        ]
        for operation in invalid:
            with self.subTest(operation=operation):
                result, next_id, decision = apply_playbook_lifecycle(BASE, [ADD, operation], 8, 999)
                self.assertEqual((result, next_id), (BASE, 8))
                self.assertEqual(decision['status'], 'rejected')
                self.assertEqual(decision['reason'], 'invalid_operations')

    def test_target_overlap_rejected_independent_of_order(self):
        for operations in ([UPDATE, MERGE], [MERGE, UPDATE], [UPDATE, UPDATE],
                           [DELETE, dict(UPDATE, bullet_id='str-00003')]):
            with self.subTest(operations=operations):
                self.assertEqual(apply_playbook_lifecycle(BASE, operations, 8, 999)[:2], (BASE, 8))
                with self.assertRaises(ValueError):
                    prepare_playbook_candidate(BASE, operations, 8)

    def test_ambiguous_base_and_id_collision_rejected(self):
        for base, next_id in ((BASE + BASE, 8), (BASE, 1), (BASE, True),
                              (BASE + '## OTHERS\n', 8)):
            with self.subTest(next_id=next_id):
                result = apply_playbook_lifecycle(base, [ADD], next_id, 999)
                self.assertEqual(result[:2], (base, next_id))
                self.assertEqual(result[2]['status'], 'rejected')

    def test_explicit_general_and_custom_whitespace_sections_work(self):
        for header in ('GENERAL', 'CUSTOM  SECTION', 'CUSTOM _ SECTION'):
            with self.subTest(header=header):
                base = '## ' + header + '\n[str-00001] helpful=1 harmful=0 :: Old advice.\n'
                result, next_id, decision = apply_playbook_lifecycle(base,
                    [dict(UPDATE, content='Updated advice.'), dict(ADD, section=header)], 8, 999)
                self.assertEqual(decision['status'], 'accepted')
                self.assertEqual(next_id, 9)
                self.assertIn('Updated advice.', result)
                self.assertIn(ADD['content'], result)
                self.assertTrue(result.startswith('## ' + header + '\n'))
        rejected = apply_playbook_lifecycle(BASE, [dict(ADD, section='general')], 8, 999)
        self.assertEqual(rejected[:2], (BASE, 8))
        self.assertEqual(rejected[2]['status'], 'rejected')

    def test_candidate_is_frozen_and_does_not_retain_mutable_proposals(self):
        operation = dict(UPDATE)
        candidate = prepare_playbook_candidate(BASE, [operation], 8)
        operation['content'] = 'changed after preparation'
        self.assertIn(UPDATE['content'], candidate.playbook)
        self.assertNotIn(operation['content'], candidate.playbook)
        with self.assertRaises(FrozenInstanceError):
            candidate.next_id = 999

    def test_exact_budget_accepts_and_over_budget_rejects_atomically(self):
        candidate = prepare_playbook_candidate(BASE, [UPDATE, ADD], 8)
        size = count_tokens(candidate.playbook)
        self.assertEqual(publish_playbook_candidate(candidate, BASE, 8, size, enforce=True)[:2],
                         (candidate.playbook, 9))
        rejected = publish_playbook_candidate(candidate, BASE, 8, size - 1, enforce=True)
        self.assertEqual(rejected[:2], (BASE, 8))
        self.assertEqual(rejected[2]['reason'], 'candidate_exceeds_budget')
        for budget in (-1, None, True):
            with self.subTest(budget=budget):
                self.assertEqual(publish_playbook_candidate(candidate, BASE, 8, budget, True)[:2], (BASE, 8))

    def test_literal_token_markers_survive_update_and_exact_budget_publication(self):
        for marker in ('<|endoftext|>', '<|fim_prefix|>'):
            with self.subTest(marker=marker):
                candidate = prepare_playbook_candidate(BASE, [dict(UPDATE, content='Treat ' + marker + ' as literal text.')], 8)
                size = count_tokens(candidate.playbook)
                result = publish_playbook_candidate(candidate, BASE, 8, size, enforce=True)
                self.assertEqual(result[:2], (candidate.playbook, 8))
                self.assertEqual(result[2]['status'], 'accepted')

    def test_empty_candidate_preserves_bytes_and_validates_budget(self):
        candidate = prepare_playbook_candidate(BASE, [], 8)
        self.assertEqual(candidate.playbook, BASE)
        self.assertEqual(publish_playbook_candidate(candidate, BASE, 8, 0, True)[:2], (BASE, 8))
        self.assertEqual(publish_playbook_candidate(candidate, BASE, 8, -1, True)[2]['status'], 'rejected')

    def test_stale_publish_and_rollback_do_not_overwrite_newer_state(self):
        candidate = prepare_playbook_candidate(BASE, [UPDATE], 8)
        stale = BASE + 'new state\n'
        self.assertEqual(publish_playbook_candidate(candidate, stale, 8, 999)[:2], (stale, 8))
        self.assertEqual(publish_playbook_candidate(candidate, BASE, 9, 999)[2]['reason'], 'stale_base')
        self.assertEqual(rollback_playbook_candidate(candidate, candidate.playbook, 8)[:2], (BASE, 8))
        self.assertEqual(rollback_playbook_candidate(candidate, stale, 9)[:2], (stale, 9))
        self.assertEqual(rollback_playbook_candidate(candidate, stale, 9)[2]['reason'], 'stale_candidate')


class CuratorLifecycleTest(unittest.TestCase):
    def setUp(self):
        from ace.core.curator import Curator
        self.curator = Curator(None, 'offline', 'synthetic')
        self.logs = tempfile.TemporaryDirectory()
        self.addCleanup(self.logs.cleanup)
        self.fields = dict(current_playbook=BASE, recent_reflection='reflection',
                           question_context='context', current_step=1, total_samples=1,
                           token_budget=999, playbook_stats={}, log_dir=self.logs.name)

    def test_gt_and_no_gt_curator_and_proposals_enable_operations_explicitly(self):
        for use_gt in (True, False):
            for method in ('curate', 'propose_operations'):
                with self.subTest(method=method, use_gt=use_gt):
                    response = json.dumps({'reasoning': 'synthetic', 'operations': [DELETE]})
                    with patch('ace.core.curator.timed_llm_call', return_value=(response, {})) as call:
                        result = getattr(self.curator, method)(**self.fields,
                            use_ground_truth=use_gt, enable_playbook_lifecycle=True,
                            **({'next_global_id': 8} if method == 'curate' else {}))
                    self.assertIn('DELETE', call.call_args.args[3])
                    self.assertIn('ground truth answers' if use_gt else 'environment feedback', call.call_args.args[3])
                    if method == 'curate':
                        self.assertNotIn('str-00003', result[0])
                        self.assertEqual(result[1], 8)
                        self.assertEqual(result[3]['playbook_lifecycle']['status'], 'accepted')
                    else:
                        self.assertEqual(result[0], [DELETE])

    def test_curator_publishes_valid_batch_and_reports_invalid_target_atomically(self):
        for operations, expected in (([UPDATE, DELETE], 'accepted'),
                                     ([ADD, dict(UPDATE, bullet_id='missing')], 'rejected')):
            with self.subTest(expected=expected):
                response = json.dumps({'reasoning': 'synthetic', 'operations': operations})
                with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
                    result = self.curator.curate(**self.fields, next_global_id=8, enable_playbook_lifecycle=True)
                self.assertEqual(result[3]['playbook_lifecycle']['status'], expected)
                if expected == 'rejected':
                    self.assertEqual(result[:3], (BASE, 8, []))
                else:
                    self.assertIn(UPDATE['content'], result[0])
                    self.assertNotIn('str-00003', result[0])

    def test_empty_model_response_preserves_state_without_a_log_directory(self):
        for method in ('curate', 'propose_operations', 'aggregate_operations'):
            fields = dict(self.fields, log_dir=None)
            if method == 'aggregate_operations':
                fields.pop('recent_reflection'); fields.pop('question_context')
                fields['operation_groups'] = []
            if method == 'curate':
                fields['next_global_id'] = 8
            with patch('ace.core.curator.timed_llm_call', return_value=('INCORRECT_DUE_TO_EMPTY_RESPONSE', {})):
                result = getattr(self.curator, method)(**fields, enable_playbook_lifecycle=True)
            if method == 'curate':
                self.assertEqual(result[:3], (BASE, 8, []))
            else:
                self.assertEqual(result[0], [])
            self.assertEqual(result[-1]['playbook_lifecycle']['reason'], 'empty_response')

    def test_reducer_keeps_lifecycle_operations_without_mutating_shared_base(self):
        fields = dict(self.fields)
        fields.pop('recent_reflection'); fields.pop('question_context')
        groups = [{'group_id': 1, 'operations': [MERGE, DELETE]}]
        response = json.dumps({'reasoning': 'synthetic', 'operations': [MERGE, DELETE]})
        with patch('ace.core.curator.timed_llm_call', return_value=(response, {})) as call:
            operations, _ = self.curator.aggregate_operations(**fields, operation_groups=groups,
                enable_playbook_lifecycle=True)
        self.assertEqual(operations, [MERGE, DELETE])
        self.assertEqual(fields['current_playbook'], BASE)
        self.assertNotIn('only supports ADD', call.call_args.args[3])

    def test_conflicting_reducer_and_unsupported_proposal_are_rejected(self):
        for method, operations in (('propose_operations', [UPDATE, {'type': 'CREATE_META'}]),
                                   ('aggregate_operations', [UPDATE, MERGE])):
            fields = dict(self.fields)
            if method == 'aggregate_operations':
                fields.pop('recent_reflection'); fields.pop('question_context')
                fields['operation_groups'] = []
            response = json.dumps({'reasoning': 'synthetic', 'operations': operations})
            with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
                result = getattr(self.curator, method)(**fields, enable_playbook_lifecycle=True)
            self.assertEqual(result[0], [])
            self.assertEqual(result[1]['playbook_lifecycle']['status'], 'rejected')


class TrainingLifecycleTest(unittest.TestCase):
    def setUp(self):
        from ace.ace import ACE
        from ace.ace_batch import ACEBatch
        from ace.core.curator import Curator
        self.classes = (ACE, ACEBatch)
        self.logs = tempfile.TemporaryDirectory()
        self.addCleanup(self.logs.cleanup)
        self.models = []
        for cls in self.classes:
            model = object.__new__(cls)
            model.playbook = BASE; model.next_global_id = 8
            model.use_bulletpoint_analyzer = False; model.bulletpoint_analyzer = None
            model.generator = Mock()
            model.generator.generate.return_value = ('{"final_answer": "ok"}', [], {})
            model.reflector = Mock()
            model.reflector.reflect.return_value = ('reflection', [], {})
            model.curator = Curator(None, 'offline', 'synthetic')
            self.models.append(model)
        self.processor = Mock()
        self.processor.answer_is_correct.side_effect = lambda answer, target: answer == target
        self.task = {'question': 'synthetic question', 'context': 'context', 'target': 'ok'}

    def train(self, model, **config):
        params = model._extract_config_params(dict(enable_playbook_lifecycle=True,
            augmented_shuffling=False, batch_size=1, **config))
        with patch('ace.ace.log_bullet_usage'), patch('ace.ace_batch.log_bullet_usage'):
            if type(model) is self.classes[0]:
                return model._train_single_sample(self.task, self.processor, 'synthetic', 1, 1,
                    self.logs.name + '/usage.jsonl', self.logs.name, params, 1)
            return model._train_batch([self.task], self.processor, 1, 1,
                self.logs.name + '/usage.jsonl', self.logs.name, params, 1)[0]

    def test_actual_single_and_batch_publish_once_with_stable_id_and_tracking(self):
        response = json.dumps({'reasoning': 'synthetic', 'operations': [UPDATE, DELETE, ADD]})
        for model in self.models:
            with self.subTest(cls=type(model).__name__):
                with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
                    output = self.train(model)
                self.assertEqual(model.next_global_id, 9)
                self.assertEqual(set(bullets(model.playbook)), {'str-00001', 'str-00002', 'misc-00004', 'sai-00008'})
                self.assertEqual(model.generator.generate.call_args.kwargs['playbook'], model.playbook)
                self.assertEqual(output[2]['playbook_lifecycle']['status'], 'accepted')

    def test_actual_single_and_batch_budget_reject_before_post_generation(self):
        response = json.dumps({'reasoning': 'synthetic', 'operations': [UPDATE, ADD]})
        for model in self.models:
            with self.subTest(cls=type(model).__name__):
                with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
                    output = self.train(model, enforce_playbook_token_budget=True, playbook_token_budget=1)
                self.assertEqual((model.playbook, model.next_global_id), (BASE, 8))
                self.assertEqual(model.generator.generate.call_args.kwargs['playbook'], BASE)
                self.assertEqual(output[2]['playbook_lifecycle']['reason'], 'candidate_exceeds_budget')

    def test_post_generation_failure_rolls_back_both_text_and_id(self):
        response = json.dumps({'reasoning': 'synthetic', 'operations': [MERGE, DELETE, ADD]})
        for model in self.models:
            def generate(**kwargs):
                if kwargs['call_id'].endswith('post_curate'):
                    raise RuntimeError('synthetic post-generation failure')
                return '{"final_answer": "ok"}', [], {}
            model.generator.generate.side_effect = generate
            with self.subTest(cls=type(model).__name__):
                with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                        self.train(model)
                self.assertEqual((model.playbook, model.next_global_id), (BASE, 8))

    def test_batch_continued_api_failure_rolls_back_counter_updates_and_tracks_rollback(self):
        model = self.models[1]
        model.reflector.reflect.return_value = ('reflection', [{'id': 'str-00001', 'tag': 'helpful'}], {})
        def generate(**kwargs):
            if kwargs['call_id'].endswith('post_curate'):
                raise RuntimeError('synthetic model failure')
            return '{"final_answer": "ok"}', [], {}
        model.generator.generate.side_effect = generate
        response = json.dumps({'reasoning': 'synthetic', 'operations': [MERGE, ADD]})
        with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
            output = self.train(model, continue_on_llm_error=True)
        self.assertEqual((model.playbook, model.next_global_id), (BASE, 8))
        self.assertEqual(output[2]['playbook_lifecycle']['rollback']['status'], 'rolled_back')

    def test_batch_sdk_failure_continues_and_rolls_back_without_opt_in_error_flag(self):
        import openai
        model = self.models[1]
        def generate(**kwargs):
            if kwargs['call_id'].endswith('post_curate'):
                raise openai.APIConnectionError(request=Mock())
            return '{"final_answer": "ok"}', [], {}
        model.generator.generate.side_effect = generate
        response = json.dumps({'reasoning': 'synthetic', 'operations': [DELETE, ADD]})
        with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
            output = self.train(model)
        self.assertEqual((model.playbook, model.next_global_id), (BASE, 8))
        self.assertEqual(output[2]['playbook_lifecycle']['rollback']['status'], 'rolled_back')
        self.assertFalse(output[2]['post_train_result']['is_correct'])

    def test_batch_continued_failure_stale_rollback_preserves_newer_state(self):
        model = self.models[1]
        newer = BASE + '# External newer state\n'
        def generate(**kwargs):
            if kwargs['call_id'].endswith('post_curate'):
                model.playbook, model.next_global_id = newer, 20
                raise RuntimeError('synthetic model failure after newer state')
            return '{"final_answer": "ok"}', [], {}
        model.generator.generate.side_effect = generate
        response = json.dumps({'reasoning': 'synthetic', 'operations': [UPDATE, ADD]})
        with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
            output = self.train(model, continue_on_llm_error=True)
        self.assertEqual((model.playbook, model.next_global_id), (newer, 20))
        self.assertEqual(output[2]['playbook_lifecycle']['rollback'],
                         {'status': 'rejected', 'reason': 'stale_candidate'})
        self.assertEqual(output[2]['post_train_result']['playbook_num_tokens'], count_tokens(newer))

    def test_batch_reducer_rejection_reports_conflict_and_keeps_base(self):
        model = self.models[1]
        response = lambda ops: json.dumps({'reasoning': 'synthetic', 'operations': ops})
        with patch('ace.core.curator.timed_llm_call', side_effect=[(response([UPDATE]), {}),
                                                               (response([UPDATE, MERGE]), {})]):
            output = self.train(model)
        self.assertEqual((model.playbook, model.next_global_id), (BASE, 8))
        self.assertEqual(output[2]['playbook_lifecycle']['status'], 'rejected')

    def test_batch_invalid_proposal_is_reported_without_publishing_partial_add(self):
        model = self.models[1]
        response = json.dumps({'reasoning': 'synthetic', 'operations': [ADD, {'type': 'CREATE_META'}]})
        with patch('ace.core.curator.timed_llm_call', return_value=(response, {})):
            output = self.train(model)
        self.assertEqual((model.playbook, model.next_global_id), (BASE, 8))
        self.assertEqual(output[2]['playbook_lifecycle']['status'], 'rejected')
        self.assertEqual(output[2]['playbook_lifecycle']['reason'], 'invalid_proposals')

    def test_seeded_constructor_allocates_after_existing_ids(self):
        for cls, module in zip(self.classes, ('ace.ace', 'ace.ace_batch')):
            with self.subTest(cls=cls.__name__):
                with patch(module + '.initialize_clients', return_value=(None, None, None)):
                    model = cls('offline', 'synthetic', 'synthetic', 'synthetic', initial_playbook=BASE)
                self.assertEqual(model.next_global_id, 5)
                candidate = prepare_playbook_candidate(model.playbook, [UPDATE, ADD], model.next_global_id)
                self.assertIn('sai-00005', candidate.playbook)
                self.assertEqual(candidate.next_id, 6)

    def test_defaults_and_analyzer_incompatibility_are_explicit(self):
        for model in self.models:
            self.assertFalse(model._extract_config_params({})['enable_playbook_lifecycle'])
            with self.assertRaisesRegex(ValueError, 'BulletpointAnalyzer'):
                model._extract_config_params({'enable_playbook_lifecycle': True, 'use_bulletpoint_analyzer': True})
