import hashlib
import json
import re
import sys
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional


class DataProcessor:
    """
    Processor for Frontier-CS tasks.

    Note: Frontier-CS is open-ended. Default correctness checks validate
    output format rather than solution quality.
    For algorithmic track, this can optionally call the Frontier-CS judge
    to score code submissions.
    """

    def __init__(
        self,
        task_name: str,
        frontier_root: Optional[str] = None,
        judge_url: str = "http://localhost:8081",
        backend: Optional[str] = None,
        use_judge: bool = True,
    ):
        """
        Initialize the data processor.

        Args:
            task_name: "algorithmic" or "research"
            frontier_root: Path to Frontier-CS repo (used to import evaluator)
            judge_url: Algorithmic judge server URL
            backend: "docker" or "skypilot" (optional)
            use_judge: Enable judge-based evaluation for algorithmic track (whether get score or just format validity)
        """
        self.task_name = task_name
        self.frontier_root = Path(frontier_root).expanduser() if frontier_root else None
        self.judge_url = judge_url
        self.backend = backend
        self.use_judge = use_judge

        self._evaluator = None
        self._evaluator_error = None
        self._score_cache: Dict[tuple, float] = {}
        self._cache_lock = threading.Lock()

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert raw JSONL items into ACE's standard format.
        """
        processed_data = []

        for item in raw_data:
            context = item.get("context", "")
            target = item.get("target", "")
            metadata = item.get("metadata", {})

            if self.task_name == "algorithmic":
                problem_id = metadata.get("problem_id")
                if problem_id is None and self.use_judge:
                    raise ValueError("Missing problem_id in metadata for algorithmic judging.")
                if problem_id is not None and (not target or (isinstance(target, str) and not target.strip())):
                    target = json.dumps(
                        {"problem_id": int(problem_id)},
                        separators=(",", ":"),
                    )

            processed_item = {
                "context": context,
                "question": self._build_question(),
                "target": target,
                "others": {
                    "original_context": context,
                    "task": self.task_name,
                    "metadata": metadata,
                },
            }
            processed_data.append(processed_item)

        return processed_data

    def _build_question(self) -> str:
        if self.task_name == "algorithmic":
            return (
                "Solve the algorithmic problem in the context. "
                "Write a C++17 program that follows the input/output format. "
                "Return only code, no explanations. Optimize for score when applicable."
            )
        # if self.task_name == "research":
        #     return (
        #         "Solve the research problem in the context. "
        #         "Implement the required Python API (e.g., Solution.solve). "
        #         "Return only code, no explanations."
        #     )
        raise ValueError(f"Unknown task: {self.task_name}")

    def _algorithmic_answer_is_valid(self, predicted: str) -> bool:
        if not predicted or not predicted.strip():
            return False
        return bool(re.search(r"\bint\s+main\s*\(", predicted)) and "#include" in predicted

    def _extract_problem_id(self, ground_truth: str) -> Optional[int]:
        if not ground_truth:
            return None
        if isinstance(ground_truth, str):
            try:
                payload = json.loads(ground_truth)
                if isinstance(payload, dict) and "problem_id" in payload:
                    return int(payload["problem_id"])
            except json.JSONDecodeError:
                if ground_truth.isdigit():
                    return int(ground_truth)
        return None

    def _ensure_evaluator(self) -> None:
        if not self.use_judge or self.task_name != "algorithmic":
            return
        if self._evaluator or self._evaluator_error:
            return
        try:
            if self.frontier_root:
                src_path = self.frontier_root / "src"
                if src_path.exists():
                    sys.path.insert(0, str(src_path))
            from frontier_cs import SingleEvaluator  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import-time errors
            self._evaluator_error = exc
            return

        self._evaluator = SingleEvaluator(
            backend=self.backend,
            base_dir=self.frontier_root,
            judge_url=self.judge_url,
        )

    def _cache_key(self, problem_id: int, code: str) -> tuple:
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        return (str(problem_id), code_hash, self.backend or "")

    def _score_algorithmic(self, predicted: str, problem_id: int) -> float:
        if not self._algorithmic_answer_is_valid(predicted):
            return 0.0

        key = self._cache_key(problem_id, predicted)
        with self._cache_lock:
            if key in self._score_cache:
                return self._score_cache[key]

        self._ensure_evaluator()
        if not self._evaluator:
            raise RuntimeError(
                "Frontier-CS evaluator not available. "
                "Set --frontier_root to the Frontier-CS repo and ensure dependencies are installed."
            )

        result = self._evaluator.evaluate(
            "algorithmic",
            problem_id=problem_id,
            code=predicted,
            backend=self.backend,
        )

        score = 0.0
        if result.success:
            score = result.score if result.score is not None else 0.0

        with self._cache_lock:
            self._score_cache[key] = score

        return score

    # def _research_answer_is_valid(self, predicted: str) -> bool:
    #     if not predicted or not predicted.strip():
    #         return False
    #     return "class Solution" in predicted and re.search(r"\bdef\s+solve\s*\(", predicted) is not None

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Format-based correctness for open-ended tasks.
        """
        if self.task_name == "algorithmic":
            # When using judge scoring, treat correctness as format validity.
            return self._algorithmic_answer_is_valid(predicted)
        # if self.task_name == "research":
        #     return self._research_answer_is_valid(predicted)

        raise ValueError(f"Unknown task: {self.task_name}")

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """
        Compute accuracy or average score.
        """
        if len(out) != len(target):
            raise ValueError("Input lists 'out' and 'target' must have the same length.")

        if not out:
            return 0.0

        if self.task_name == "algorithmic" and self.use_judge:
            scores = []
            for predicted, ground_truth in zip(out, target):
                problem_id = self._extract_problem_id(ground_truth)
                if problem_id is None:
                    scores.append(0.0)
                    continue
                scores.append(self._score_algorithmic(predicted, problem_id))
            return sum(scores) / len(scores) if scores else 0.0

        correct_count = 0
        for predicted, ground_truth in zip(out, target):
            if self.answer_is_correct(predicted, ground_truth):
                correct_count += 1

        return correct_count / len(out)

