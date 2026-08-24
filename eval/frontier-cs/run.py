#!/usr/bin/env python3
"""
Example usage script for Frontier-CS tasks with ACE.
"""
import os
import json
import argparse

try:
    from .data_processor import DataProcessor
except ImportError:
    from data_processor import DataProcessor
from ace import ACE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ACE System - Frontier-CS")

    # Task configuration
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=["algorithmic"],
        help="Task name ('algorithmic')",
    )
    parser.add_argument(
        "--initial_playbook_path",
        type=str,
        default=None,
        help="Path to initial playbook (optional)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "online", "eval_only"],
        help="Run mode",
    )

    # Model configuration
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai"],
                        help="API provider")
    parser.add_argument("--generator_model", type=str, default="DeepSeek-V3.1",
                        help="Model for generator")
    parser.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1",
                        help="Model for reflector")
    parser.add_argument("--curator_model", type=str, default="DeepSeek-V3.1",
                        help="Model for curator")

    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_num_rounds", type=int, default=3,
                        help="Max reflection rounds for incorrect answers")
    parser.add_argument("--curator_frequency", type=int, default=1,
                        help="Run curator every N steps")
    parser.add_argument("--eval_steps", type=int, default=20,
                        help="Evaluate every N steps")
    parser.add_argument("--online_eval_frequency", type=int, default=10,
                        help="Update playbook every N samples for evaluation in online mode")
    parser.add_argument("--save_steps", type=int, default=10,
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

    # Frontier-CS evaluator configuration (algorithmic track)
    parser.add_argument(
        "--frontier_root",
        type=str,
        default="/Users/jingzhuohu/repos/Frontier-CS",
        help="Path to Frontier-CS repo",
    )
    parser.add_argument(
        "--judge_url",
        type=str,
        default="http://localhost:8081",
        help="Algorithmic judge server URL",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["docker", "skypilot"],
        help="Evaluation backend (default: track-specific)",
    )
    parser.add_argument(
        "--no_judge",
        action="store_true",
        help="Disable judge-based evaluation (falls back to format checks)",
    )
    # Output configuration
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")

    return parser.parse_args()


def load_data(data_path: str):
    """Load JSONL data file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def preprocess_data(config, mode, processor):
    """
    Load train/val/test data for the specified task.
    """
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")

    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print("ACE SYSTEM - FRONTIER-CS")
    print(f"{'=' * 60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'=' * 60}\n")

    with open("./eval/frontier-cs/data/sample_config.json", "r") as f:
        task_config = json.load(f)

    data_processor = DataProcessor(
        task_name=args.task_name,
        frontier_root=args.frontier_root,
        judge_url=args.judge_url,
        backend=args.backend,
        use_judge=not args.no_judge,
    )

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        task_config[args.task_name], args.mode, data_processor
    )

    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
    )

    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "initial_playbook_path": args.initial_playbook_path,
        "api_provider": args.api_provider,
    }

    ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config,
    )


if __name__ == "__main__":
    main()

