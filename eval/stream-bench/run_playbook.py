#!/usr/bin/env python3
"""
Script to run a playbook from intermediate_playbooks folder on processed data
and evaluate accuracy.

Usage:
    python run_playbook.py --results_dir <path> --playbook_file <filename>

Example:
    python run_playbook.py --results_dir results/ace_run_20260119_234301_bird_all_hard_to_easy_online --playbook_file window_4_final_playbook.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add stream-bench to path
stream_bench_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, stream_bench_dir)

from ace.core import Generator
from utils import initialize_clients
from data_processor import DataProcessor


def load_playbook(playbook_path: str) -> str:
    """Load playbook content from file."""
    with open(playbook_path, 'r') as f:
        return f.read()




def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from generator response.

    The response may be:
    1. JSON format with reasoning and final_answer
    2. Plain SQL query

    Args:
        response: Generator response

    Returns:
        Extracted SQL query
    """
    # Try to parse as JSON first
    try:
        response_json = json.loads(response)
        # Try to get final_answer field
        if 'final_answer' in response_json:
            sql = response_json['final_answer']
            # Remove any markdown code blocks
            sql = sql.replace('```sql', '').replace('```', '').strip()
            return sql
        # If no final_answer, try to get the whole response
        return response.strip()
    except (json.JSONDecodeError, TypeError):
        # If not JSON, assume it's plain SQL
        # Remove any markdown code blocks
        sql = response.replace('```sql', '').replace('```', '').strip()
        return sql


def generate_predictions_parallel(
    generator: Generator,
    samples: List[Dict[str, Any]],
    playbook: str,
    num_workers: int = 4
) -> tuple[List[str], Dict[str, Any]]:
    """
    Generate predictions for all samples using the playbook in parallel.

    Args:
        generator: Generator instance
        samples: List of samples with context and question
        playbook: Playbook content to use
        num_workers: Number of parallel workers

    Returns:
        Tuple of (predictions, error_stats)
        - predictions: List of predicted SQL queries
        - error_stats: Dictionary with error statistics
    """
    predictions = [None] * len(samples)
    error_info = []  # Track detailed error information

    def process_sample(idx: int, sample: Dict[str, Any]) -> tuple:
        """Process a single sample and return (idx, prediction, error_type)."""
        try:
            context = sample['context']
            question = sample['question']

            # Generate prediction using playbook
            # Generator.generate returns (response, bullet_ids, call_info)
            response, bullet_ids, call_info = generator.generate(
                question=question,
                playbook=playbook,
                context=context,
                reflection="(empty)",  # Explicitly pass empty reflection to minimize tokens
                use_json_mode=True  # Use JSON mode to get structured response
            )

            # Extract SQL from response
            predicted_sql = extract_sql_from_response(response)

            # Check if this was an error response from timed_llm_call
            error_type = None
            if "INCORRECT_DUE_TO_EMPTY_RESPONSE" in predicted_sql:
                error_type = "empty_response"
            elif "INCORRECT_DUE_TO_INVALID_PROMPT" in predicted_sql:
                error_type = "invalid_prompt"
            elif call_info.get('error'):
                # Check for context length exceeded
                error_msg = call_info.get('error', '')
                if 'context_length_exceeded' in error_msg or 'tokens exceed' in error_msg:
                    error_type = "context_length_exceeded"
                else:
                    error_type = "api_error"

            return idx, predicted_sql, error_type
        except Exception as e:
            error_str = str(e)
            print(f"Error processing sample {idx}: {error_str}")

            # Classify error type
            error_type = "unknown_error"
            if 'context_length_exceeded' in error_str or 'tokens exceed' in error_str:
                error_type = "context_length_exceeded"
            elif 'timeout' in error_str.lower() or 'timed out' in error_str.lower():
                error_type = "timeout"
            elif 'rate limit' in error_str.lower() or '429' in error_str:
                error_type = "rate_limit"
            elif '400' in error_str or 'invalid_prompt' in error_str.lower():
                error_type = "client_error"
            elif '500' in error_str or 'server error' in error_str.lower():
                error_type = "server_error"

            # Return a placeholder SQL that will fail evaluation
            return idx, "SELECT 1", error_type

    print(f"\nGenerating predictions with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_sample, i, sample): i
            for i, sample in enumerate(samples)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            idx, prediction, error_type = future.result()
            predictions[idx] = prediction

            if error_type:
                error_info.append({
                    'sample_idx': idx,
                    'error_type': error_type,
                    'question': samples[idx].get('question', '')[:100]  # First 100 chars
                })

            completed += 1

            if completed % 10 == 0 or completed == len(samples):
                print(f"  Progress: {completed}/{len(samples)} samples completed")

    # Generate error statistics
    error_stats = {
        'total_errors': len(error_info),
        'error_breakdown': {},
        'error_details': error_info
    }

    # Count errors by type
    for error in error_info:
        error_type = error['error_type']
        error_stats['error_breakdown'][error_type] = error_stats['error_breakdown'].get(error_type, 0) + 1

    if error_info:
        print(f"\n⚠️  Warning: {len(error_info)} samples failed during generation")
        print("Error breakdown:")
        for error_type, count in sorted(error_stats['error_breakdown'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count}")

    return predictions, error_stats


def evaluate_test_samples(
    predictions: List[str],
    test_samples: List[Dict[str, Any]],
    bird_db_root: str
) -> Dict[str, Any]:
    """
    Evaluate predictions using test_samples.json (which already has ground truth).

    Args:
        predictions: List of predicted SQL queries
        test_samples: List of test samples with ground truth SQL
        bird_db_root: Path to BIRD database root

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n" + "="*70)
    print("EVALUATING ON TEST SAMPLES")
    print("="*70)
    print(f"Total test samples: {len(test_samples)}")
    print("="*70)

    if len(predictions) != len(test_samples):
        print(f"\nError: Mismatch between predictions ({len(predictions)}) and samples ({len(test_samples)})")
        return {
            'accuracy': 0.0,
            'total_samples': 0,
            'correct': 0,
            'error': 'Prediction count mismatch'
        }

    # Initialize DataProcessor for evaluation
    data_processor = DataProcessor(bird_db_root=bird_db_root)

    # Evaluate each sample
    correct = 0
    results = []

    # Track results by difficulty
    difficulty_stats = {
        'simple': {'correct': 0, 'total': 0},
        'moderate': {'correct': 0, 'total': 0},
        'challenging': {'correct': 0, 'total': 0}
    }

    print("\nEvaluating predictions...")
    for i, (pred, sample) in enumerate(zip(predictions, test_samples)):
        try:
            gt = sample['target']
            meta = sample.get('others', {})
            difficulty = meta.get('difficulty', 'unknown')

            is_correct = data_processor.answer_is_correct(pred, gt, meta)

            if is_correct:
                correct += 1

            # Track by difficulty
            if difficulty in difficulty_stats:
                difficulty_stats[difficulty]['total'] += 1
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1

            results.append({
                'question': sample['question'],
                'db_name': meta.get('db_name', ''),
                'difficulty': difficulty,
                'predicted_sql': pred,
                'ground_truth_sql': gt,
                'is_correct': is_correct
            })
        except Exception as e:
            print(f"  Error evaluating sample {i}: {e}")
            meta = sample.get('others', {})
            difficulty = meta.get('difficulty', 'unknown')

            # Track failed sample by difficulty
            if difficulty in difficulty_stats:
                difficulty_stats[difficulty]['total'] += 1

            results.append({
                'question': sample.get('question', ''),
                'db_name': meta.get('db_name', ''),
                'difficulty': difficulty,
                'predicted_sql': pred,
                'ground_truth_sql': sample.get('target', ''),
                'is_correct': False,
                'error': str(e)
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(test_samples):
            print(f"  Progress: {i + 1}/{len(test_samples)} samples evaluated (correct: {correct})")

    accuracy = correct / len(test_samples) if len(test_samples) > 0 else 0.0

    # Calculate difficulty-specific accuracies
    difficulty_accuracies = {}
    for diff, stats in difficulty_stats.items():
        if stats['total'] > 0:
            difficulty_accuracies[diff] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }

    return {
        'accuracy': accuracy,
        'total_samples': len(test_samples),
        'correct': correct,
        'difficulty_breakdown': difficulty_accuracies,
        'results': results
    }




def load_run_config(results_dir: str) -> Dict[str, Any]:
    """Load run_config.json from results directory."""
    config_path = os.path.join(results_dir, 'run_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Run a playbook on processed data and evaluate accuracy'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to results directory (e.g., results/ace_run_20260119_234301_bird_all_hard_to_easy_online)'
    )
    parser.add_argument(
        '--playbook_file',
        type=str,
        default=None,
        help='Playbook file path relative to results_dir (e.g., intermediate_playbooks/window_4_final_playbook.txt). If not provided, runs initial evaluation with empty playbook.'
    )
    parser.add_argument(
        '--bird_db_root',
        type=str,
        default='eval/stream-bench/data/bird/dev_databases',
        help='Path to BIRD database root directory (for SQL execution during evaluation)'
    )
    parser.add_argument(
        '--api_provider',
        type=str,
        default=None,
        choices=['sambanova', 'together', 'openai'],
        help='API provider for LLM calls (defaults to value from run_config.json)'
    )
    parser.add_argument(
        '--generator_model',
        type=str,
        default=None,
        help='Model name for generator (defaults to generator_model from run_config.json)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers for generation'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Optional output file to save detailed results (JSON)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which dataset to evaluate on: train_samples.json, val_samples.json, or test_samples.json (default: test)'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    # Handle playbook file (optional)
    playbook_path = None
    if args.playbook_file:
        # Join playbook_file with results_dir
        playbook_path = os.path.join(args.results_dir, args.playbook_file)
        if not os.path.exists(playbook_path):
            print(f"Error: Playbook file not found: {playbook_path}")
            return 1

    # Load run config to get default model and API provider
    run_config = load_run_config(args.results_dir)

    # Use config values if args not provided
    if args.api_provider is None:
        args.api_provider = run_config.get('config', {}).get('api_provider') or run_config.get('api_provider', 'sambanova')
        print(f"Using API provider from run_config.json: {args.api_provider}")

    if args.generator_model is None:
        args.generator_model = run_config.get('generator_model', 'DeepSeek-V3.1')
        print(f"Using generator_model from run_config.json: {args.generator_model}")

    # Get bird_db_root from config if not provided via CLI
    # Hardcode database paths: train/val use train_databases, test uses dev_databases
    if args.bird_db_root == 'eval/stream-bench/data/bird/dev_databases':  # Using default
        if args.dataset in ['train', 'val']:
            args.bird_db_root = 'eval/stream-bench/data/bird_train/train_databases'
        else:  # test
            args.bird_db_root = 'eval/stream-bench/data/bird/dev_databases'
        print(f"Using bird_db_root for {args.dataset} dataset: {args.bird_db_root}")

    # Load playbook (or use empty for initial evaluation)
    if playbook_path:
        print(f"\nLoading playbook from: {playbook_path}")
        playbook = load_playbook(playbook_path)
        print(f"Playbook loaded ({len(playbook)} characters)")
    else:
        print(f"\nNo playbook provided - running INITIAL EVALUATION with empty playbook")
        playbook = ""

    # Load samples from processed_data (has everything we need)
    samples_filename = f'{args.dataset}_samples.json'
    samples_path = os.path.join(args.results_dir, 'processed_data', samples_filename)

    if not os.path.exists(samples_path):
        print(f"\nError: {args.dataset.capitalize()} samples file not found: {samples_path}")
        print("This file should be created during the ACE training run.")
        return 1

    print(f"\nLoading {args.dataset} samples from: {samples_path}")
    print(f"  (This file contains the {args.dataset} data with ground truth SQL)")
    with open(samples_path, 'r') as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} {args.dataset} samples")

    # Initialize generator
    print(f"\nInitializing generator with {args.api_provider} API...")
    generator_client, _, _ = initialize_clients(args.api_provider)
    generator = Generator(generator_client, args.api_provider, args.generator_model, max_tokens=4096)

    # Generate predictions
    predictions, error_stats = generate_predictions_parallel(
        generator, samples, playbook, num_workers=args.num_workers
    )

    # Evaluate
    print(f"\nEvaluating predictions using execution-based evaluation...")
    eval_results = evaluate_test_samples(predictions, samples, args.bird_db_root)

    # Print results
    print("\n" + "="*70)
    if args.playbook_file:
        print(f"EVALUATION RESULTS - {args.dataset.upper()} DATASET")
        print("="*70)
        print(f"Playbook: {args.playbook_file}")
    else:
        print(f"INITIAL EVALUATION RESULTS - {args.dataset.upper()} DATASET (empty playbook)")
        print("="*70)
        print(f"Playbook: <empty>")
    print(f"Dataset: {args.dataset}_samples.json")
    print(f"\nOverall Performance:")
    print(f"  Total samples evaluated: {eval_results['total_samples']}")
    print(f"  Correct: {eval_results['correct']}")
    print(f"  Accuracy: {eval_results['accuracy']:.2%}")

    # Print difficulty breakdown if available
    if 'difficulty_breakdown' in eval_results and eval_results['difficulty_breakdown']:
        print(f"\nPerformance by Difficulty:")
        for difficulty in ['simple', 'moderate', 'challenging']:
            if difficulty in eval_results['difficulty_breakdown']:
                stats = eval_results['difficulty_breakdown'][difficulty]
                print(f"  {difficulty.capitalize():12s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    if error_stats['total_errors'] > 0:
        print(f"\nAPI Errors:")
        print(f"  Total errors during generation: {error_stats['total_errors']}")
        print(f"  (These samples were marked as incorrect)")
    print("="*70)

    # Save detailed results if requested
    if args.output_file:
        # Save output file under the results_dir directory
        output_path = os.path.join(args.results_dir, args.output_file)

        # Warn if file already exists
        if os.path.exists(output_path):
            print(f"\nWarning: Output file already exists and will be overwritten: {output_path}")

        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump({
                    'dataset': args.dataset,
                    'dataset_file': f'{args.dataset}_samples.json',
                    'playbook_file': args.playbook_file if args.playbook_file else '<empty>',
                    'playbook_path': playbook_path if playbook_path else None,
                    'is_initial_evaluation': args.playbook_file is None,
                    'bird_db_root': args.bird_db_root,
                    'accuracy': eval_results['accuracy'],
                    'total_samples': eval_results['total_samples'],
                    'correct': eval_results['correct'],
                    'difficulty_breakdown': eval_results.get('difficulty_breakdown', {}),
                    'api_errors': error_stats['total_errors'],
                    'error_breakdown': error_stats['error_breakdown'],
                    'error_details': error_stats['error_details'],
                    'results': eval_results['results']
                }, f, indent=2)
            print(f"\nDetailed results saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving results to {output_path}: {e}")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
