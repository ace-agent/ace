import os
import re
import sqlite3
import time
from typing import List, Dict, Any, Optional, Tuple

class DataProcessor:
    """
    DataProcessor for BIRD and CoSQL with thread-safe evaluation.

    Evaluation mode: execute predicted & gold on sqlite DB and compare result sets

    Thread-safety: db_name metadata flows through sample['others'] dict to enable
    safe parallel evaluation across multiple workers.
    """

    def __init__(
        self,
        bird_db_root: Optional[str] = None,
        cosql_db_root: Optional[str] = None,
        exec_timeout_ms: int = 20000,
        exec_max_rows: int = 20000,
        max_samples: Optional[int] = None,  # None = use all samples
        db_name: Optional[str] = None,  # None = use mixed databases (no filter)
        difficulty_filter: Optional[str] = None,  # None = no difficulty filtering
        curriculum: Optional[str] = None,  # None = no curriculum ordering
        task: str = "bird",  # "bird" or "cosql"
    ):
        """
        Initialize DataProcessor.

        Args:
            bird_db_root: Root directory for BIRD databases
            cosql_db_root: Root directory for CoSQL databases
            task: Task type ("bird" or "cosql")
            difficulty_filter: Strategy for selecting samples by difficulty (dataset-level). Options:
                - None: No filtering, use all samples
                - "simple-only": Only simple difficulty samples
                - "moderate-only": Only moderate difficulty samples
                - "challenging-only": Only challenging difficulty samples
                - "balanced": Equal distribution (1/3 from each difficulty)
                - "quasi_balanced": Equal distribution with fallback to closest difficulty.
                  If not enough samples: simple uses moderate, challenging uses moderate,
                  moderate uses half simple + half challenging
            curriculum: Strategy for ordering samples (run-level). Options:
                - None: No reordering (original order from dataset)
                - "easy_to_hard": Easy -> Medium -> Challenging order
                - "hard_to_easy": Challenging -> Medium -> Easy order
                - "random": Random order (fixed seed)
        """
        self.bird_db_root = bird_db_root
        self.cosql_db_root = cosql_db_root
        self.task = task
        self.exec_timeout_ms = exec_timeout_ms
        self.exec_max_rows = exec_max_rows
        self.max_samples = max_samples
        self.db_name = db_name
        self.difficulty_filter = difficulty_filter
        self.curriculum = curriculum

        # Validate difficulty_filter option
        valid_filters = [
            None, "simple-only", "moderate-only", "challenging-only", "balanced", "quasi_balanced"
        ]
        if self.difficulty_filter not in valid_filters:
            raise ValueError(
                f"Invalid difficulty_filter '{self.difficulty_filter}'. "
                f"Valid options: {[f for f in valid_filters if f is not None]}"
            )

        # Validate curriculum option
        valid_curricula = [None, "easy_to_hard", "hard_to_easy", "random"]
        if self.curriculum not in valid_curricula:
            raise ValueError(
                f"Invalid curriculum '{self.curriculum}'. "
                f"Valid options: {[c for c in valid_curricula if c is not None]}"
            )

    # -------------------------
    # REQUIRED SIGNATURES
    # -------------------------

    def process_task_data(self, raw_data):
        """
        Convert your BIRD JSONL rows into standardized format:
          [{"context": ..., "question": ..., "target": ..., "others": {...}}]

        db_name is stored in each sample's 'others' dict for thread-safe evaluation.
        """
        processed = []

        # Step 1: Filter by db_name if specified
        if self.db_name is not None:
            raw_data = [
                item for item in raw_data
                if (item.get("db_name") or item.get("db_id") or "") == self.db_name
            ]
            print(f"After db_name filter ('{self.db_name}'): {len(raw_data)} samples")

        # Step 2: Apply difficulty_filter for dataset-level selection
        if self.difficulty_filter is not None:
            raw_data = self._apply_difficulty_filter(raw_data)

        # Step 3: Apply curriculum for run-level ordering
        if self.curriculum is not None:
            raw_data = self._apply_curriculum_ordering(raw_data)

        # Step 4: Apply max_samples limit (if not already applied by balanced difficulty filter)
        if self.max_samples is not None and self.difficulty_filter != "balanced":
            if len(raw_data) > self.max_samples:
                raw_data = raw_data[:self.max_samples]
                print(f"Applied max_samples limit: {self.max_samples} samples")

        # Print summary of processed data
        self._print_data_summary(raw_data)

        for item in raw_data:
            db_name = item.get("db_name") or item.get("db_id") or ""
            question = item.get("question", "")
            target_sql = item.get("sql", "") or item.get("SQL", "")
            db_schema = item.get("db_schema", "")

            context = f"""You are given a database schema and a question.

        INSTRUCTIONS:
        - Output ONLY a valid SQL query.
        - Do NOT include explanations, comments, markdown, or any extra text.
        - You may ONLY reference tables and columns that appear in the schema below.
        - Do NOT hallucinate tables, columns, or relationships.

        DATABASE SCHEMA:
        {db_schema}
        """

            processed.append({
                "context": context,
                "question": question,
                "target": target_sql,
                "others": {
                    "question_id": item.get("question_id"),
                    "difficulty": item.get("difficulty"),
                    "db_name": db_name,
                    "task": self.task,
                    "data_source": f"streambench_{self.task}",
                    "turn_id": item.get("turn_id"),  # For CoSQL conversational turns
                }
            })

        return processed

    def answer_is_correct(self, predicted, ground_truth, sample_metadata=None, return_exec_results=False):
        """
        Compare predicted vs ground_truth using exec mode.

        Args:
            predicted: Predicted SQL query
            ground_truth: Ground truth SQL query
            sample_metadata: Optional dict containing 'db_name' and other metadata
            return_exec_results: If True, return tuple of (is_correct, exec_results_dict)

        Returns:
            bool: True if execution results match, False otherwise
            OR
            tuple: (bool, dict) if return_exec_results=True, where dict contains:
                - 'predicted_result': List of tuples from predicted SQL execution
                - 'ground_truth_result': List of tuples from ground truth SQL execution
                - 'db_name': Database name used for evaluation
                - 'error': Error message if execution failed
        """
        # Extract db_name from metadata
        db_name = ""
        if sample_metadata:
            db_name = sample_metadata.get("db_name", "")

        # If db_name not available, return False
        if not db_name:
            print(f"Warning: No db_name available in sample metadata")
            if return_exec_results:
                return False, {"error": "No db_name available in sample metadata", "db_name": ""}
            return False

        print(f"\n[EVAL START] Evaluating on DB: {db_name}")
        print(f"[EVAL START] Predicted SQL: {predicted[:200]}...")  # First 200 chars
        print(f"[EVAL START] Ground truth SQL: {ground_truth[:200]}...")

        try:
            result, exec_results = self._exec_match(predicted, ground_truth, db_name, return_exec_results=return_exec_results)
            print(f"[EVAL DONE] Result: {result}")
            if return_exec_results:
                return result, exec_results
            return result
        except FileNotFoundError:
            # Re-raise database not found errors to stop execution
            raise
        except Exception as e:
            # Other execution errors: print and return False
            print(f"[EVAL ERROR] Exception during evaluation: {e}")
            if return_exec_results:
                return False, {"error": str(e), "db_name": db_name}
            return False

    def evaluate_accuracy(self, predictions, ground_truths, samples=None):
        """
        Calculate accuracy using execution-based evaluation.

        For parallel test evaluation: The actual correctness is determined by
        answer_is_correct() in worker threads. This method re-evaluates using
        sample metadata if available, or falls back to string comparison.

        Args:
            predictions: List of predicted SQL queries
            ground_truths: List of ground truth SQL queries
            samples: Optional list of sample dicts with 'others' metadata

        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        if len(predictions) == 0:
            return 0.0

        correct = 0
        for i, (p, g) in enumerate(zip(predictions, ground_truths)):
            # If we have sample metadata, use execution-based evaluation
            if samples and i < len(samples):
                sample_metadata = samples[i].get("others", None) if isinstance(samples[i], dict) else None
                if self.answer_is_correct(p, g, sample_metadata):
                    correct += 1
            else:
                # Fallback to string comparison for training (no metadata available)
                if p.strip().lower() == g.strip().lower():
                    correct += 1

        return correct / len(predictions)

    # -------------------------
    # DIFFICULTY FILTER & CURRICULUM LOGIC
    # -------------------------

    def _apply_difficulty_filter(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply difficulty-based filtering to select samples (dataset-level).

        Args:
            raw_data: List of raw data items with 'difficulty' field

        Returns:
            Filtered list based on difficulty_filter strategy
        """
        if not self.difficulty_filter:
            return raw_data

        # Categorize samples by difficulty
        simple_samples = []
        moderate_samples = []
        challenging_samples = []
        unknown_samples = []

        for item in raw_data:
            difficulty = (item.get("difficulty") or "").lower()
            if difficulty == "simple":
                simple_samples.append(item)
            elif difficulty == "moderate":
                moderate_samples.append(item)
            elif difficulty == "challenging":
                challenging_samples.append(item)
            else:
                unknown_samples.append(item)

        # Apply difficulty filter strategy
        if self.difficulty_filter == "simple-only":
            result = simple_samples
            print(f"Difficulty filter 'simple-only': Selected {len(result)} simple samples")

        elif self.difficulty_filter == "moderate-only":
            result = moderate_samples
            print(f"Difficulty filter 'moderate-only': Selected {len(result)} moderate samples")

        elif self.difficulty_filter == "challenging-only":
            result = challenging_samples
            print(f"Difficulty filter 'challenging-only': Selected {len(result)} challenging samples")

        elif self.difficulty_filter == "balanced":
            # Equal distribution from each difficulty (1/3 each)
            # If max_samples is set, distribute it equally across difficulties
            if self.max_samples is not None:
                target_per_difficulty = self.max_samples // 3
                min_count = min(
                    len(simple_samples),
                    len(moderate_samples),
                    len(challenging_samples),
                    target_per_difficulty
                )
            else:
                min_count = min(len(simple_samples), len(moderate_samples), len(challenging_samples))

            # Check if we have samples from all difficulty levels
            if min_count == 0:
                missing = []
                if len(simple_samples) == 0:
                    missing.append("simple")
                if len(moderate_samples) == 0:
                    missing.append("moderate")
                if len(challenging_samples) == 0:
                    missing.append("challenging")
                raise ValueError(
                    f"Difficulty filter 'balanced' requires samples from all difficulty levels. "
                    f"Missing difficulty levels: {', '.join(missing)}. "
                    f"Available: simple={len(simple_samples)}, moderate={len(moderate_samples)}, "
                    f"challenging={len(challenging_samples)}"
                )

            # Combine samples without reordering (order depends on curriculum)
            result = (
                simple_samples[:min_count] +
                moderate_samples[:min_count] +
                challenging_samples[:min_count]
            )

            # Check if we can meet max_samples requirement
            if self.max_samples is not None and len(result) < self.max_samples:
                raise ValueError(
                    f"Cannot meet max_samples={self.max_samples} with difficulty filter 'balanced'. "
                    f"Need {self.max_samples // 3} samples per difficulty, but only have: "
                    f"simple={len(simple_samples)}, moderate={len(moderate_samples)}, "
                    f"challenging={len(challenging_samples)}. "
                    f"Can only provide {len(result)} samples ({min_count} of each difficulty)."
                )

            print(f"Difficulty filter 'balanced': Selected {min_count} from each difficulty "
                  f"(total: {len(result)} samples)")

        elif self.difficulty_filter == "quasi_balanced":
            # Quasi-balanced: Try to get equal distribution (1/3 each),
            # but if a difficulty doesn't have enough samples, use closest difficulty:
            # - For challenging: use moderate if not enough challenging
            # - For simple: use moderate if not enough simple
            # - For moderate: use half simple and half challenging if not enough moderate

            if self.max_samples is not None:
                target_per_difficulty = self.max_samples // 3
            else:
                # Try to get the maximum possible balanced distribution
                target_per_difficulty = max(
                    len(simple_samples),
                    len(moderate_samples),
                    len(challenging_samples)
                )

            # Collect samples for each difficulty category with fallback
            selected_simple = []
            selected_moderate = []
            selected_challenging = []

            # Simple samples: use simple first, then moderate
            if len(simple_samples) >= target_per_difficulty:
                selected_simple = simple_samples[:target_per_difficulty]
            else:
                selected_simple = simple_samples[:]
                needed = target_per_difficulty - len(selected_simple)
                # Use moderate as fallback
                selected_simple.extend(moderate_samples[:needed])

            # Challenging samples: use challenging first, then moderate
            if len(challenging_samples) >= target_per_difficulty:
                selected_challenging = challenging_samples[:target_per_difficulty]
            else:
                selected_challenging = challenging_samples[:]
                needed = target_per_difficulty - len(selected_challenging)
                # Use moderate as fallback
                selected_challenging.extend(moderate_samples[:needed])

            # Moderate samples: use moderate first, then half simple and half challenging
            if len(moderate_samples) >= target_per_difficulty:
                selected_moderate = moderate_samples[:target_per_difficulty]
            else:
                selected_moderate = moderate_samples[:]
                needed = target_per_difficulty - len(selected_moderate)
                # Split needed samples between simple and challenging
                half_needed = needed // 2
                remainder = needed % 2

                # Use simple and challenging (not already used in other categories)
                # To avoid reusing samples, we need to track what we've already taken
                simple_used = len(selected_simple) if len(simple_samples) < target_per_difficulty else 0
                challenging_used = len(selected_challenging) if len(challenging_samples) < target_per_difficulty else 0

                from_simple = simple_samples[simple_used:simple_used + half_needed + remainder]
                from_challenging = challenging_samples[challenging_used:challenging_used + half_needed]

                selected_moderate.extend(from_simple)
                selected_moderate.extend(from_challenging)

            # Combine samples
            result = selected_simple + selected_moderate + selected_challenging

            # Print detailed selection info
            simple_from_moderate = max(0, target_per_difficulty - len(simple_samples))
            challenging_from_moderate = max(0, target_per_difficulty - len(challenging_samples))
            moderate_from_others = max(0, target_per_difficulty - len(moderate_samples))

            print(f"Difficulty filter 'quasi_balanced': Target {target_per_difficulty} per difficulty")
            print(f"  Simple: {len(selected_simple)} samples "
                  f"({len(simple_samples)} native, {simple_from_moderate} from moderate)")
            print(f"  Moderate: {len(selected_moderate)} samples "
                  f"({len(moderate_samples)} native, {moderate_from_others} from simple/challenging)")
            print(f"  Challenging: {len(selected_challenging)} samples "
                  f"({len(challenging_samples)} native, {challenging_from_moderate} from moderate)")
            print(f"  Total: {len(result)} samples")

        else:
            # Should not reach here due to validation in __init__
            result = raw_data

        if unknown_samples:
            print(f"Warning: {len(unknown_samples)} samples with unknown difficulty were excluded")

        return result

    def _apply_curriculum_ordering(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply curriculum-based ordering to samples (run-level).

        Args:
            raw_data: List of raw data items with 'difficulty' field

        Returns:
            Reordered list based on curriculum strategy
        """
        if not self.curriculum:
            return raw_data

        # Categorize samples by difficulty
        simple_samples = []
        moderate_samples = []
        challenging_samples = []
        unknown_samples = []

        for item in raw_data:
            difficulty = (item.get("difficulty") or "").lower()
            if difficulty == "simple":
                simple_samples.append(item)
            elif difficulty == "moderate":
                moderate_samples.append(item)
            elif difficulty == "challenging":
                challenging_samples.append(item)
            else:
                unknown_samples.append(item)

        # Apply curriculum ordering
        if self.curriculum == "easy_to_hard":
            # Easy -> Medium -> Challenging
            result = simple_samples + moderate_samples + challenging_samples + unknown_samples
            print(f"Curriculum 'easy_to_hard': Ordered {len(simple_samples)} simple -> "
                  f"{len(moderate_samples)} moderate -> {len(challenging_samples)} challenging")

        elif self.curriculum == "hard_to_easy":
            # Challenging -> Medium -> Easy
            result = challenging_samples + moderate_samples + simple_samples + unknown_samples
            print(f"Curriculum 'hard_to_easy': Ordered {len(challenging_samples)} challenging -> "
                  f"{len(moderate_samples)} moderate -> {len(simple_samples)} simple")

        elif self.curriculum == "random":
            # Random order with fixed seed
            import random
            result = raw_data.copy()
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(result)
            print(f"Curriculum 'random': Randomly shuffled {len(result)} samples (seed=42)")

        else:
            # Should not reach here due to validation in __init__
            result = raw_data

        return result

    def _print_data_summary(self, raw_data: List[Dict[str, Any]]) -> None:
        """
        Print a detailed summary of the processed data.

        Args:
            raw_data: List of processed data items
        """
        print("\n" + "="*70)
        print("PROCESSED DATA SUMMARY")
        print("="*70)

        # Total size
        print(f"Total samples: {len(raw_data)}")

        # Database selection
        if self.db_name:
            print(f"Database filter: '{self.db_name}'")
        else:
            # Count unique databases
            db_names = set()
            for item in raw_data:
                db = item.get("db_name") or item.get("db_id") or "unknown"
                db_names.add(db)
            print(f"Database filter: None (using {len(db_names)} databases: {', '.join(sorted(db_names))})")

        # Difficulty filter (dataset-level)
        if self.difficulty_filter:
            print(f"Difficulty filter (dataset-level): {self.difficulty_filter}")
        else:
            print("Difficulty filter: None")

        # Curriculum (run-level ordering)
        if self.curriculum:
            print(f"Curriculum ordering (run-level): {self.curriculum}")
        else:
            print("Curriculum ordering: None (original order)")

        # Difficulty distribution
        from collections import Counter
        difficulty_counts = Counter()
        for item in raw_data:
            difficulty = (item.get("difficulty") or "unknown").lower()
            difficulty_counts[difficulty] += 1

        print(f"\nDifficulty distribution:")
        print(f"  Simple:      {difficulty_counts.get('simple', 0):4d} samples")
        print(f"  Moderate:    {difficulty_counts.get('moderate', 0):4d} samples")
        print(f"  Challenging: {difficulty_counts.get('challenging', 0):4d} samples")
        if difficulty_counts.get('unknown', 0) > 0:
            print(f"  Unknown:     {difficulty_counts.get('unknown', 0):4d} samples")

        # Order of difficulties (first 20 and last 20 samples)
        if len(raw_data) > 0:
            difficulties = [(item.get("difficulty") or "unknown").lower() for item in raw_data]

            print(f"\nDifficulty order:")
            if len(difficulties) <= 40:
                # Show all if 40 or fewer
                order_str = " -> ".join(difficulties)
                print(f"  {order_str}")
            else:
                # Show first 20 and last 20
                first_20 = " -> ".join(difficulties[:20])
                last_20 = " -> ".join(difficulties[-20:])
                print(f"  First 20: {first_20}")
                print(f"  ... ({len(difficulties) - 40} more samples) ...")
                print(f"  Last 20:  {last_20}")

        print("="*70 + "\n")

    # -------------------------
    # EXECUTION EVAL INTERNALS
    # -------------------------

    def _exec_match(self, predicted_sql: str, gold_sql: str, db_name: str, return_exec_results: bool = False):
        sqlite_path = self._find_sqlite_path(db_name)
        if not sqlite_path:
            # DB not found -> raise error and stop execution
            db_root = self.cosql_db_root if self.task == "cosql" else self.bird_db_root
            error_msg = f"SQLite DB for {db_name} not found under {db_root}. Please check database configuration."
            print(f"\n--- FATAL ERROR: Database Not Found ---")
            print(f"DB: {db_name}")
            print(f"Task: {self.task}")
            print(f"Expected location: {db_root}/{db_name}/{db_name}.sqlite")
            print(f"Error: {error_msg}")
            print("-" * 50)
            raise FileNotFoundError(error_msg)

        try:
            print(f"[EXEC] Running PREDICTED SQL on {db_name}")
            pred_res = self._run_sql(sqlite_path, predicted_sql)

            print(f"[EXEC] Running GROUND TRUTH SQL on {db_name}")
            gold_res = self._run_sql(sqlite_path, gold_sql)

            # Print execution results
            print(f"\n--- Execution Results ---")
            print(f"DB: {db_name}")
            print(f"\nPredicted SQL:\n{predicted_sql}")
            print(f"\nPredicted Result ({len(pred_res)} rows):")
            for row in pred_res[:10]:  # Print first 10 rows
                print(f"  {row}")
            if len(pred_res) > 10:
                print(f"  ... ({len(pred_res) - 10} more rows)")

            print(f"\nGround Truth SQL:\n{gold_sql}")
            print(f"\nGround Truth Result ({len(gold_res)} rows):")
            for row in gold_res[:10]:  # Print first 10 rows
                print(f"  {row}")
            if len(gold_res) > 10:
                print(f"  ... ({len(gold_res) - 10} more rows)")
            print("-" * 50)

            # Compare results
            print(f"[EXEC] Normalizing and comparing results...")
            match = self._normalize_result(pred_res) == self._normalize_result(gold_res)
            print(f"[EXEC] Match result: {match}")

            if return_exec_results:
                exec_results = {
                    "predicted_result": pred_res,
                    "ground_truth_result": gold_res,
                    "db_name": db_name
                }
                return match, exec_results
            return match, {}

        except Exception as e:
            print(f"\n--- Execution Error ---")
            print(f"DB: {db_name}")
            print(f"Error: {e}")
            print("-" * 50)
            if return_exec_results:
                return False, {"error": str(e), "db_name": db_name}
            return False, {}

    def _find_sqlite_path(self, db_name: str) -> Optional[str]:
        """
        Find SQLite database path for either BIRD or CoSQL.

        Typical layouts:
          BIRD: <bird_db_root>/<db_name>/<db_name>.sqlite
          CoSQL: <cosql_db_root>/<db_name>/<db_name>.sqlite
        """
        # Determine which db_root to use based on task
        if self.task == "cosql":
            db_root = self.cosql_db_root
        else:
            db_root = self.bird_db_root

        if not db_root:
            return None

        # Try standard layout: <db_root>/<db_name>/<db_name>.sqlite
        p = os.path.join(db_root, db_name, f"{db_name}.sqlite")
        if os.path.exists(p):
            return p

        return None

    def _run_sql(self, sqlite_path: str, sql: str) -> List[Tuple[Any, ...]]:
        sql = (sql or "").strip().rstrip(";")
        if not sql:
            raise ValueError("Empty SQL")

        # guard against writes/DDL in evaluation
        lowered = sql.lower()
        if any(k in lowered for k in ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "pragma ", "attach "]):
            raise ValueError("Unsafe SQL in evaluation")

        print(f"[DEBUG] Connecting to database: {sqlite_path}")
        conn = sqlite3.connect(sqlite_path, timeout=self.exec_timeout_ms / 1000.0)

        # Set up timeout mechanism using progress handler
        start_time = time.time()
        timeout_seconds = self.exec_timeout_ms / 1000.0

        def progress_handler():
            if time.time() - start_time > timeout_seconds:
                print(f"[TIMEOUT] Query exceeded {timeout_seconds}s timeout")
                return 1  # Non-zero return aborts the operation
            return 0

        try:
            # Call progress handler every 1000 VM instructions
            conn.set_progress_handler(progress_handler, 1000)

            print(f"[DEBUG] Executing SQL query...")
            cur = conn.cursor()
            cur.execute(sql)

            print(f"[DEBUG] Fetching results (max {self.exec_max_rows} rows)...")
            rows = cur.fetchmany(self.exec_max_rows)

            elapsed = time.time() - start_time
            print(f"[DEBUG] Query completed in {elapsed:.2f}s, returned {len(rows)} rows")

            return [tuple(r) for r in rows]
        except sqlite3.OperationalError as e:
            elapsed = time.time() - start_time
            if "interrupted" in str(e).lower():
                raise TimeoutError(f"SQL query timed out after {elapsed:.2f}s (limit: {timeout_seconds}s)")
            raise
        finally:
            conn.close()

    @staticmethod
    def _normalize_result(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
        """
        Normalize query results for comparison.
        - Rounds floats to 6 decimal places
        - Sorts rows for order-independent comparison
        """
        def norm(v):
            if isinstance(v, float):
                return round(v, 6)
            return v

        normed = [tuple(norm(v) for v in row) for row in rows]
        return sorted(normed)
