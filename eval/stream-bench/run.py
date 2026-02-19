#!/usr/bin/env python3
"""
Stream Bench task runner using ACE system.
"""
import os
import sys
import json
import re
import time
import argparse
import traceback

from ace import ACE
from .data_processor import DataProcessor
from .plot import plot_online_performance, plot_training_progress, plot_offline_training_progress


def load_data(data_path: str):
    """
    Load and process data from a JSONL file.

    Args:
        data_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def parse_args():
    """Parse command line arguments for stream-bench."""
    parser = argparse.ArgumentParser(description='ACE System - Stream Bench')

    # Task configuration
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task (e.g., 'finer', 'formula')")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Path to initial playbook (optional)")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode: 'offline' for offline training with validation, "
                             "'online' for online training and testing on test split, "
                             "'eval_only' for testing only with provided playbook")

    # Model configuration
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai"], help="API provider")
    parser.add_argument("--generator_model", type=str,
                        default="DeepSeek-V3.1",
                        help="Model for generator")
    parser.add_argument("--reflector_model", type=str,
                        default="DeepSeek-V3.1",
                        help="Model for reflector")
    parser.add_argument("--curator_model", type=str,
                        default="DeepSeek-V3.1",
                        help="Model for curator")

    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_num_rounds", type=int, default=3,
                        help="Max reflection rounds for incorrect answers")
    parser.add_argument("--curator_frequency", type=int, default=1,
                        help="Run curator every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--online_eval_frequency", type=int, default=15,
                        help="Update playbook every N samples for evaluation in online mode")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save intermediate playbooks every N steps")

    # System configuration
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens for LLM responses")
    parser.add_argument("--playbook_token_budget", type=int, default=80000,
                        help="Total token budget for playbook")
    parser.add_argument("--test_workers", type=int, default=20,
                        help="Number of parallel workers for testing")

    # Prompt configuration
    parser.add_argument("--json_mode", action="store_true",
                        help="Enable JSON mode for LLM calls")
    parser.add_argument("--no_ground_truth", action="store_true",
                        help="Don't use ground truth in reflection")

    # Bulletpoint analyzer configuration
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true",
                        help="Enable bulletpoint analyzer for deduplication and merging")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90,
                        help="Similarity threshold for bulletpoint analyzer (0-1, default: 0.90)")

    # SQL evaluation configuration
    parser.add_argument("--pass_sql_eval_results", action="store_true",
                        help="Pass SQL execution results to reflector for better error analysis")

    # Output configuration
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")

    # Stream-bench specific arguments
    parser.add_argument("--data_config", type=str, required=True,
                        help="Path to data configuration JSON file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate performance plot for online mode (shows accuracy vs steps)")
    parser.add_argument("--db_name", type=str, default=None,
                        help="Database name to filter data (optional, overrides config)")
    parser.add_argument("--curriculum", type=str, default=None, choices=["easy_to_hard", "hard_to_easy", "random"],
                        help="Curriculum ordering strategy: easy_to_hard, hard_to_easy, random")

    return parser.parse_args()


class TeeLogger:
    """Logger that writes to both terminal and file simultaneously with auto-flush."""

    def __init__(self, log_file_path, mode='w'):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, mode, buffering=1)  # Line buffering
        self.log_file_path = log_file_path

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        # Force flush to ensure immediate write
        self.terminal.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.terminal
        self.log_file.close()
        

def preprocess_data(task_name, config, mode, db_name=None, curriculum=None):
    """
    Load training and test data for the specified task.

    Args:
        task_name: Name of the task
        config: Configuration dictionary with data paths and settings
        mode: Run mode ('offline', 'online', 'online', or 'eval_only')
        db_name: Database name from command line args
        curriculum: Curriculum ordering from command line args

    Returns:
        Tuple of (train_samples, val_samples, test_samples, train_processor, val_processor, test_processor)
        - For offline mode: all three sample sets and all three processors are returned
        - For online/eval_only mode: only test_samples and test_processor (train/val processors are None)
    """
    # Get max_samples from config
    # max_samples serves as default for all splits
    # Individual limits (max_train_samples, max_val_samples, max_test_samples) override the default
    max_samples = config.get("max_samples", None)
    max_train_samples = config.get("max_train_samples") if "max_train_samples" in config else max_samples
    max_val_samples = config.get("max_val_samples") if "max_val_samples" in config else max_samples
    max_test_samples = config.get("max_test_samples") if "max_test_samples" in config else max_samples

    # Detect task type based on config keys
    # If cosql_db_root is present, it's a CoSQL task, otherwise BIRD
    if "cosql_db_root" in config:
        task = "cosql"
        # Get cosql_db_root from config
        cosql_db_root = config.get("cosql_db_root", "eval/stream-bench/data/cosql")
        bird_db_root = None
        bird_train_db_root = None
        bird_val_db_root = None
        bird_test_db_root = None
        cosql_train_db_root = cosql_db_root
        cosql_val_db_root = cosql_db_root
        cosql_test_db_root = cosql_db_root

        print(f"[CONFIG] Task: CoSQL")
        print(f"[CONFIG] Database path: {cosql_db_root}")
    else:
        task = "bird"
        # Get bird_db_root from config, with support for separate train/val/test database paths
        # bird_db_root serves as default for all splits
        # Individual paths (bird_train_db_root, bird_val_db_root, bird_test_db_root) override the default
        bird_db_root = config.get("bird_db_root", "eval/stream-bench/data/bird/dev_databases")
        bird_train_db_root = config.get("bird_train_db_root") if "bird_train_db_root" in config else bird_db_root
        bird_val_db_root = config.get("bird_val_db_root") if "bird_val_db_root" in config else bird_db_root
        bird_test_db_root = config.get("bird_test_db_root") if "bird_test_db_root" in config else bird_db_root
        cosql_db_root = None
        cosql_train_db_root = None
        cosql_val_db_root = None
        cosql_test_db_root = None

        print(f"[CONFIG] Task: BIRD")
        print(f"[CONFIG] Database paths:")
        print(f"  bird_train_db_root: {bird_train_db_root}")
        print(f"  bird_val_db_root: {bird_val_db_root}")
        print(f"  bird_test_db_root: {bird_test_db_root}")

    # Get difficulty_filter from config (dataset-level selection)
    difficulty_filter = config.get("difficulty_filter", None)

    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        # Create processor for test data
        test_processor = DataProcessor(
            bird_db_root=bird_test_db_root,
            cosql_db_root=cosql_test_db_root,
            task=task,
            max_samples=max_test_samples,
            db_name=db_name,
            difficulty_filter=difficulty_filter,
            curriculum=curriculum
        )

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = test_processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")

        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")

        return train_samples, val_samples, test_samples, None, None, test_processor

    # For offline mode, load train, val, and optionally test data
    else:
        # Create separate processors for train, val, and test to apply different max_samples
        train_processor = DataProcessor(
            bird_db_root=bird_train_db_root,
            cosql_db_root=cosql_train_db_root,
            task=task,
            max_samples=max_train_samples,
            db_name=db_name,
            difficulty_filter=difficulty_filter,
            curriculum=curriculum
        )

        val_processor = DataProcessor(
            bird_db_root=bird_val_db_root,
            cosql_db_root=cosql_val_db_root,
            task=task,
            max_samples=max_val_samples,
            db_name=db_name,
            difficulty_filter=difficulty_filter,
            curriculum=curriculum
        )

        test_processor = DataProcessor(
            bird_db_root=bird_test_db_root,
            cosql_db_root=cosql_test_db_root,
            task=task,
            max_samples=max_test_samples,
            db_name=db_name,
            difficulty_filter=difficulty_filter,
            curriculum=curriculum
        )

        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = train_processor.process_task_data(train_samples)
        val_samples = val_processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = test_processor.process_task_data(test_samples)
        else:
            test_samples = []

        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")

        # Return all three processors for proper evaluation of each split
        return train_samples, val_samples, test_samples, train_processor, val_processor, test_processor


def main():
    """Main execution function."""
    # Start total timing
    total_start_time = time.time()

    args = parse_args()

    # Create temporary log directory to capture all output from the start
    temp_log_dir = os.path.join(args.save_path, "temp_logs")
    os.makedirs(temp_log_dir, exist_ok=True)
    log_timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_log_path = os.path.join(temp_log_dir, f"terminal_output_{log_timestamp}.txt")

    # Set up logger immediately to capture ALL output
    logger = TeeLogger(temp_log_path)
    sys.stdout = logger

    # Print initial banner (now captured by logger)
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM - Stream Bench")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"Data Config: {args.data_config}")
    print(f"Logging all terminal output to: {temp_log_path}")
    print(f"{'='*60}\n")

    try:

        # Load data configuration
        with open(args.data_config, 'r') as f:
            data_config = json.load(f)

        # Get task-specific config
        if args.task_name not in data_config:
            raise ValueError(f"Task '{args.task_name}' not found in config file: {args.data_config}")

        task_config = data_config[args.task_name]

        # Print config settings for max_samples
        max_samples_default = task_config.get("max_samples", None)
        has_overrides = "max_train_samples" in task_config or "max_val_samples" in task_config or "max_test_samples" in task_config

        if has_overrides:
            # Show overrides with default fallback
            print(f"Max samples (from config):")
            if max_samples_default is not None:
                print(f"  - Default: {max_samples_default}")
            train_val = task_config.get('max_train_samples', max_samples_default or 'No limit')
            val_val = task_config.get('max_val_samples', max_samples_default or 'No limit')
            test_val = task_config.get('max_test_samples', max_samples_default or 'No limit')
            print(f"  - Train: {train_val}")
            print(f"  - Validation: {val_val}")
            print(f"  - Test: {test_val}")
        elif max_samples_default is not None:
            # Only default specified
            print(f"Max samples (from config): {max_samples_default} (applies to all splits)")
        else:
            print(f"Max samples: No limit")

        if args.db_name:
            print(f"Database filter: {args.db_name}")
        else:
            print(f"Database filter: None (using mixed databases)")

        if "difficulty_filter" in task_config:
            print(f"Difficulty filter (from config): {task_config['difficulty_filter']}")
        else:
            print(f"Difficulty filter: None (no filtering)")

        if args.curriculum:
            print(f"Curriculum ordering: {args.curriculum}")
        else:
            print(f"Curriculum ordering: None (original order)")

        print()  # blank line

        train_samples, val_samples, test_samples, train_processor, val_processor, test_processor = preprocess_data(
            args.task_name,
            task_config,
            args.mode,
            db_name=args.db_name,
            curriculum=args.curriculum
        )

        # Load initial playbook (or use empty if None provided)
        initial_playbook = load_initial_playbook(args.initial_playbook_path)
        if initial_playbook:
            print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
        else:
            print("Using empty playbook as initial playbook\n")

        # Create ACE system with a custom wrapper to intercept path creation
        ace_system = ACE(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.reflector_model,
            curator_model=args.curator_model,
            max_tokens=args.max_tokens,
            initial_playbook=initial_playbook,
            use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
            bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold
        )

        # Extract config filename (without extension) from config path
        config_filename = os.path.splitext(os.path.basename(args.data_config))[0]

        # Prepare configuration
        config = {
            'num_epochs': args.num_epochs,
            'max_num_rounds': args.max_num_rounds,
            'curator_frequency': args.curator_frequency,
            'eval_steps': args.eval_steps,
            'online_eval_frequency': args.online_eval_frequency,
            'save_steps': args.save_steps,
            'playbook_token_budget': args.playbook_token_budget,
            'task_name': args.task_name,
            'mode': args.mode,
            'json_mode': args.json_mode,
            'no_ground_truth': args.no_ground_truth,
            'save_dir': args.save_path,  # Pass parent directory
            'test_workers': args.test_workers,
            'initial_playbook_path': args.initial_playbook_path,
            'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
            'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
            'pass_sql_eval_results': args.pass_sql_eval_results,
            'api_provider': args.api_provider,
            'config_name': config_filename,
            'db_name': args.db_name,
            'curriculum': args.curriculum
        }

        # Create a save hook to intercept when ACE creates the save path
        original_setup_paths = ace_system._setup_paths
        run_save_path_container = {'path': None}

        def setup_paths_with_data_save(*args, **kwargs):
            """Wrapper that saves processed data right after path creation."""
            result = original_setup_paths(*args, **kwargs)
            # Extract save_path from result (first element of tuple)
            save_path = result[0] if isinstance(result, tuple) else result
            run_save_path_container['path'] = save_path

            # Save processed data immediately after folder creation
            print(f"\nSaving preprocessed data to: {save_path}")
            processed_data_dir = os.path.join(save_path, "processed_data")
            os.makedirs(processed_data_dir, exist_ok=True)

            if train_samples is not None:
                train_path = os.path.join(processed_data_dir, "train_samples.json")
                with open(train_path, 'w') as f:
                    json.dump(train_samples, f, indent=2)
                print(f"  - Saved train samples ({len(train_samples)} samples)")

            if val_samples is not None:
                val_path = os.path.join(processed_data_dir, "val_samples.json")
                with open(val_path, 'w') as f:
                    json.dump(val_samples, f, indent=2)
                print(f"  - Saved val samples ({len(val_samples)} samples)")

            if test_samples is not None:
                test_path = os.path.join(processed_data_dir, "test_samples.json")
                with open(test_path, 'w') as f:
                    json.dump(test_samples, f, indent=2)
                print(f"  - Saved test samples ({len(test_samples)} samples)")

            print()  # blank line
            return result

        # Replace the method temporarily
        ace_system._setup_paths = setup_paths_with_data_save

        # Execute using the unified run method
        print(f"Starting ACE run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        run_start_time = time.time()

        results = ace_system.run(
            mode=args.mode,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            train_processor=train_processor,
            val_processor=val_processor,
            test_processor=test_processor,
            config=config
        )

        run_elapsed_time = time.time() - run_start_time
        print(f"\nACE run completed in {run_elapsed_time/60:.2f} minutes ({run_elapsed_time:.2f} seconds)")

        # Get the actual save path that was created
        run_save_path = run_save_path_container['path']
        if not run_save_path:
            # Fallback to results if something went wrong with the hook
            run_save_path = results.get('save_path', args.save_path)

        # Extract timestamp from the ace_run folder name to match it exactly
        # Folder format: ace_run_YYYYMMDD_HHMMSS_task_name_...
        folder_name = os.path.basename(run_save_path)
        timestamp_match = re.search(r'ace_run_(\d{8}_\d{6})', folder_name)
        if timestamp_match:
            ace_run_timestamp = timestamp_match.group(1)
        else:
            # Fallback to original timestamp if extraction fails
            ace_run_timestamp = log_timestamp

        # Move the log file from temp location to final location
        final_log_path = os.path.join(run_save_path, f"terminal_output_{ace_run_timestamp}.txt")

        # Close current logger before moving file
        logger.close()

        # Move the log file to final location
        import shutil
        shutil.move(temp_log_path, final_log_path)

        # Reopen logger with final path in APPEND mode to continue logging
        logger = TeeLogger(final_log_path, mode='a')
        sys.stdout = logger

        print(f"\nMoved terminal output log to: {final_log_path}")

        # Calculate and display total timing
        total_elapsed_time = time.time() - total_start_time

        print(f"\n{'='*60}")
        print(f"TOTAL EXECUTION TIME")
        print(f"{'='*60}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_elapsed_time/60:.2f} minutes ({total_elapsed_time:.2f} seconds)")
        print(f"ACE run time: {run_elapsed_time/60:.2f} minutes ({run_elapsed_time:.2f} seconds)")
        print(f"{'='*60}\n")

        # Generate performance plots if requested
        if args.plot:
            print(f"\n{'='*60}")
            print(f"GENERATING PERFORMANCE PLOTS")
            print(f"{'='*60}\n")

            if args.mode == 'online':
                plot_online_performance(run_save_path, args.mode)
                plot_training_progress(run_save_path, args.mode)
            elif args.mode == 'offline':
                plot_offline_training_progress(run_save_path)
            else:
                print(f"Skipping plot generation - not available for {args.mode} mode")

        # Close the logger
        if logger:
            # Print before closing since we're using the logger
            final_message = f"Terminal output saved to {final_log_path}"
            logger.close()
            # Print to terminal after logger is closed
            print(final_message)

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: An exception occurred")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise
    finally:
        # Ensure logger is closed even if there's an error
        if 'logger' in locals() and logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
