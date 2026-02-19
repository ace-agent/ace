#!/usr/bin/env python3
"""
Example usage script for the ACE system.

"""
import os
import json
import openai
import argparse
from datetime import datetime
from .data_processor import DataProcessor

from ace import ACE
from utils import initialize_clients

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ACE System - Refactored')
    
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
    
    # Output configuration
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")
    
    return parser.parse_args()

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

def preprocess_data(task_name, config, mode):
    """
    Load training and test data for the specified task.
    
    Args:
        task_name: Name of the task
        config: Configuration dictionary with data paths
        mode: Run mode ('offline', 'online', or 'eval_only')
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples, data_processor)
        - For offline mode: all three are loaded
        - For online mode: only test_samples
        - For eval_only mode: only test_samples
    """
    processor = DataProcessor(task_name=task_name)
    
    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")
        
        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")
    
    # For offline mode, load train, val, and optionally test data
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
        
        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")
    
    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def main():
    """Main execution function."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'='*60}\n")
    
    # Load data
    with open("./eval/finance/data/sample_config.json", 'r') as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, 
        task_config[args.task_name],
        args.mode
    )
        
    # Load initial playbook (or use empty if None provided)
    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")
    
    # Create ACE system
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
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
        'api_provider': args.api_provider
    }
    
    # Execute using the unified run method
    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config
    )
        

if __name__ == "__main__":
    main()