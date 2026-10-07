import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from ace.ace import ACE
from ace.ace_batch import ACEBatch
from ace.core.curator import Curator
from playbook_utils import (
    apply_curator_operations,
    apply_curator_operations_with_budget,
)
from utils import count_tokens


class PlaybookBudgetTest(unittest.TestCase):
    def setUp(self):
        network = patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        self.playbook = "## STRATEGIES & INSIGHTS\n"
        self.operation = {
            "type": "ADD",
            "section": "STRATEGIES & INSIGHTS",
            "content": "A concise synthetic budget test bullet.",
        }
        self.operations = [self.operation]
        self.candidate, self.next_id = apply_curator_operations(
            self.playbook, self.operations, 7
        )

    def test_exact_boundary_is_accepted_without_changing_id_policy(self):
        budget = count_tokens(self.candidate)
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, self.operations, 7, budget, enforce=True
        )
        self.assertEqual(result, self.candidate)
        self.assertEqual(next_id, self.next_id)
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["base_tokens"], count_tokens(self.playbook))
        self.assertEqual(decision["candidate_tokens"], budget)
        self.assertEqual(decision["budget_tokens"], budget)
        self.assertEqual(decision["reason"], "within_budget")

    def test_over_budget_returns_original_playbook_and_does_not_consume_id(self):
        budget = count_tokens(self.candidate) - 1
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, self.operations, 7, budget, enforce=True
        )
        self.assertEqual(result, self.playbook)
        self.assertEqual(next_id, 7)
        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(decision["reason"], "candidate_exceeds_budget")
        self.assertGreater(decision["candidate_tokens"], decision["budget_tokens"])

    def test_empty_operations_are_accepted_and_do_not_allocate_id(self):
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, [], 7, 0, enforce=True
        )
        self.assertEqual(result, self.playbook)
        self.assertEqual(next_id, 7)
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["reason"], "no_operations")

    def test_disabled_enforcement_preserves_existing_behavior(self):
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, self.operations, 7, 0, enforce=False
        )
        self.assertEqual(result, self.candidate)
        self.assertEqual(next_id, self.next_id)
        self.assertEqual(decision["enforced"], False)
        self.assertEqual(decision["reason"], "enforcement_disabled")

    def test_curator_rejection_is_structured_and_state_unchanged(self):
        with tempfile.TemporaryDirectory() as log_dir:
            response = json.dumps({"reasoning": "synthetic", "operations": self.operations})
            budget = count_tokens(self.candidate) - 1
            with patch("ace.core.curator.timed_llm_call", return_value=(response, {})):
                result = Curator(None, "offline", "test").curate(
                    current_playbook=self.playbook,
                    recent_reflection="reflection",
                    question_context="context",
                    current_step=1,
                    total_samples=1,
                    token_budget=budget,
                    playbook_stats={},
                    call_id="budget-test",
                    log_dir=log_dir,
                    next_global_id=7,
                    enforce_playbook_budget=True,
                )
        self.assertEqual(result[0], self.playbook)
        self.assertEqual(result[1], 7)
        self.assertEqual(result[2], [])
        self.assertEqual(result[3]["playbook_budget"]["status"], "rejected")
        self.assertEqual(result[3]["playbook_budget"]["base_tokens"], count_tokens(self.playbook))

    def test_ace_and_batch_default_to_compatibility(self):
        self.assertFalse(ACE._extract_config_params(object.__new__(ACE), {})[
            "enforce_playbook_token_budget"
        ])
        self.assertFalse(ACEBatch._extract_config_params(object.__new__(ACEBatch), {})[
            "enforce_playbook_token_budget"
        ])
        self.assertTrue(ACE._extract_config_params(
            object.__new__(ACE), {"enforce_playbook_token_budget": True}
        )["enforce_playbook_token_budget"])
        self.assertTrue(ACEBatch._extract_config_params(
            object.__new__(ACEBatch), {"enforce_playbook_token_budget": True}
        )["enforce_playbook_token_budget"])

    def role_response(self, response, method="curate"):
        fields = dict(
            current_playbook=self.playbook, current_step=1, total_samples=1,
            token_budget=2000, playbook_stats={},
        )
        if method == "aggregate_operations":
            fields["operation_groups"] = [{"operations": self.operations}]
        else:
            fields.update(recent_reflection="reflection", question_context="context")
        if method == "curate":
            fields.update(next_global_id=7, enforce_playbook_budget=True)
        with tempfile.TemporaryDirectory() as log_dir:
            with patch("ace.core.curator.timed_llm_call", return_value=(response, {})):
                return getattr(Curator(None, "offline", "test"), method)(
                    **fields, log_dir=log_dir
                )

    def malformed_responses(self):
        responses = ["not JSON", "null", "[]", json.dumps({
            "reasoning": "synthetic", "operations": None,
        })]
        for field in ("section", "content", "type"):
            for value in (None, 3, {}, []):
                responses.append(json.dumps({
                    "reasoning": "synthetic",
                    "operations": [self.operation, dict(self.operation, **{field: value})],
                }))
        return responses

    def test_malformed_responses_reject_every_operation_before_publication(self):
        for response in self.malformed_responses():
            for method in ("curate", "propose_operations", "aggregate_operations"):
                with self.subTest(response=response, method=method):
                    result = self.role_response(response, method)
                    expected = (self.playbook, 7, [], {}) if method == "curate" else ([], {})
                    self.assertEqual(result, expected)

    def test_unsupported_only_helper_preserves_exact_text_and_id(self):
        original = self.playbook + "\n\n"
        operations = [{"type": kind} for kind in ("DELETE", "UPDATE", "MERGE", "REMOVE")]
        for enforce in (False, True):
            with self.subTest(enforce=enforce):
                result, next_id, decision = apply_curator_operations_with_budget(
                    original, operations, 7, 0, enforce=enforce
                )
                self.assertEqual((result, next_id), (original, 7))
                self.assertEqual(decision["candidate_tokens"], count_tokens(original))
                self.assertEqual(decision["reason"],
                                 "no_operations" if enforce else "enforcement_disabled")

    def test_invalid_budget_is_rejected_even_without_executable_operations(self):
        for operations in ([], [{"type": "DELETE"}], self.operations):
            for budget in (None, True, -1, 2.5, "100"):
                with self.subTest(operations=operations, budget=budget):
                    with self.assertRaises(ValueError):
                        apply_curator_operations_with_budget(
                            self.playbook, operations, 7, budget, enforce=True
                        )

    def batch_response(self, response, budget, enforce=True):
        batch = object.__new__(ACEBatch)
        batch.playbook = self.playbook
        batch.next_global_id = 7
        batch.use_bulletpoint_analyzer = False
        batch.bulletpoint_analyzer = None
        batch.curator = Curator(None, "offline", "test")
        batch.generator = Mock()
        batch.generator.generate.return_value = (json.dumps({"final_answer": "answer"}), [], {})
        task = {"context": "context", "question": "question", "target": "answer"}
        batch._generate_and_reflect_single_sample = Mock(return_value={
            "task_dict": task, "pre_train_answer": "answer", "final_answer": "answer",
            "is_correct": True, "reflection_content": "reflection", "all_bullet_tags": [],
            "context": "context", "question": "question", "target": "answer",
            "step_id": "train_s_1", "tracking_dict": {},
        })
        processor = Mock()
        processor.answer_is_correct.side_effect = lambda answer, target: answer == target
        config = batch._extract_config_params({
            "playbook_token_budget": budget, "enforce_playbook_token_budget": enforce,
            "augmented_shuffling": False, "curator_num_groups": 1,
        })
        proposal = json.dumps({"reasoning": "synthetic", "operations": self.operations})
        with tempfile.TemporaryDirectory() as log_dir:
            with patch("ace.core.curator.timed_llm_call", side_effect=[(proposal, {}), (response, {})]) as call:
                output = batch._train_batch(
                    batch=[task], data_processor=processor, batch_step_start=1, epoch=1,
                    usage_log_path=f"{log_dir}/usage.jsonl", log_dir=log_dir,
                    config_params=config, total_samples=1,
                )
        self.assertEqual(call.call_count, 2)
        self.assertEqual(batch.generator.generate.call_args.kwargs["playbook"], batch.playbook)
        self.assertEqual(output[0][:2], ("answer", "answer"))
        return batch.playbook, batch.next_global_id, output[0][2]

    def test_real_batch_reducer_publishes_only_within_budget_or_when_disabled(self):
        response = json.dumps({"reasoning": "synthetic", "operations": self.operations})
        candidate_tokens = count_tokens(self.candidate)
        for budget, enforce, expected, expected_id, status in (
            (candidate_tokens - 1, True, self.playbook, 7, "rejected"),
            (candidate_tokens, True, self.candidate, self.next_id, "accepted"),
            (0, False, self.candidate, self.next_id, "accepted"),
        ):
            with self.subTest(budget=budget, enforce=enforce):
                result, next_id, tracking = self.batch_response(response, budget, enforce)
                self.assertEqual((result, next_id), (expected, expected_id))
                self.assertEqual(tracking["playbook_budget"]["status"], status)
                self.assertEqual(tracking["playbook_budget"]["candidate_tokens"], candidate_tokens)
                self.assertEqual(tracking["post_train_result"]["playbook_num_tokens"], count_tokens(expected))

    def test_real_batch_reducer_invalid_or_unsupported_responses_keep_state(self):
        unsupported = json.dumps({"reasoning": "synthetic", "operations": [
            {"type": kind} for kind in ("DELETE", "UPDATE", "MERGE", "REMOVE")
        ]})
        for response in self.malformed_responses() + [unsupported]:
            with self.subTest(response=response):
                result, next_id, tracking = self.batch_response(response, 2000)
                self.assertEqual((result, next_id), (self.playbook, 7))
                self.assertEqual(tracking["playbook_budget"]["reason"], "no_operations")

    def test_literal_token_markers_are_budgeted_as_ordinary_text(self):
        markers = ("<|endoftext|>", "<|fim_prefix|>")
        for marker in markers:
            with self.subTest(marker=marker):
                self.assertGreater(count_tokens(marker), 1)
                self.playbook = f"## STRATEGIES & INSIGHTS\n[old-00001] helpful=0 harmful=0 :: Literal {marker}\n"
                self.operation["content"] = f"Preserve {marker} when inspecting text."
                self.candidate, self.next_id = apply_curator_operations(
                    self.playbook, self.operations, 7
                )
                budget = count_tokens(self.candidate)
                response = json.dumps({"reasoning": "synthetic", "operations": self.operations})
                for enforce, limit, expected, expected_id in (
                    (True, budget, self.candidate, self.next_id),
                    (True, budget - 1, self.playbook, 7),
                    (False, 0, self.candidate, self.next_id),
                ):
                    result, next_id, tracking = self.batch_response(response, limit, enforce)
                    self.assertEqual((result, next_id), (expected, expected_id))
                    self.assertEqual(tracking["playbook_budget"]["candidate_tokens"], budget)
                result, next_id, _ = apply_curator_operations_with_budget(
                    self.playbook, [], 7, 0, enforce=True
                )
                self.assertEqual((result, next_id), (self.playbook, 7))

    def test_real_ace_rejected_candidate_is_not_used_by_post_curator_generation(self):
        ace = object.__new__(ACE)
        ace.playbook, ace.next_global_id = self.playbook, 7
        ace.use_bulletpoint_analyzer, ace.bulletpoint_analyzer = False, None
        ace.curator = Curator(None, "offline", "test")
        ace.generator = Mock()
        ace.generator.generate.return_value = (json.dumps({"final_answer": "answer"}), [], {})
        ace.reflector = Mock()
        ace.reflector.reflect.return_value = ("reflection", [], {})
        processor = Mock()
        processor.answer_is_correct.side_effect = lambda answer, target: answer == target
        config = ace._extract_config_params({
            "playbook_token_budget": count_tokens(self.candidate) - 1,
            "enforce_playbook_token_budget": True,
        })
        response = json.dumps({"reasoning": "synthetic", "operations": self.operations})
        with tempfile.TemporaryDirectory() as log_dir:
            with patch("ace.core.curator.timed_llm_call", return_value=(response, {})):
                output = ace._train_single_sample(
                    task_dict={"context": "context", "question": "question", "target": "answer"},
                    data_processor=processor, step_id="train_s_1", epoch=1, step=1,
                    usage_log_path=f"{log_dir}/usage.jsonl", log_dir=log_dir,
                    config_params=config, total_samples=1,
                )
        self.assertEqual((ace.playbook, ace.next_global_id), (self.playbook, 7))
        self.assertEqual(ace.generator.generate.call_args.kwargs["playbook"], self.playbook)
        self.assertEqual(output[2]["curator_result"]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
