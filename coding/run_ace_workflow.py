#!/usr/bin/env python3
"""
Complete ACE workflow for Python Code Generation (MBPP dataset).

This script runs ACE training which includes:
- Initial evaluation (baseline, no playbook) on test set
- Training to improve the prompt/playbook
- Final evaluation with the best playbook

Results are saved in training/ace_run.../initial_test_results.json and final_test_results.json

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    cd /path/to/prompt_optimisation_gepa_ace
    source .venv/bin/activate
    cd ace
    python -m coding.run_ace_workflow

Options:
    --max-train N       Limit training samples (for faster testing)
    --max-test N        Limit test samples (for faster testing)
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory (ace/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('/Users/marie/Documents/github/config.json', 'r') as f:
    config = json.load(f)

    for key in config:
        os.environ[key] = config[key]


def log(message: str = "", end: str = "\n"):
    """Print with timestamp and ACE prefix for easy log identification."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[ACE {timestamp}] {message}", end=end, flush=True)


def save_report(results_dir, results_summary, best_playbook, args, n_train, n_val, n_test):
    """Save a human-readable markdown report with all results."""
    
    report_path = os.path.join(results_dir, "REPORT.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate improvement
    baseline = results_summary.get('baseline_accuracy')
    final = results_summary.get('final_accuracy')
    improvement = results_summary.get('improvement')
    
    with open(report_path, 'w') as f:
        f.write("# ACE Python Code Generation (MBPP) - Run Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        
        # Configuration
        f.write("## Configuration\n\n")
        f.write(f"- **API Provider:** {args.api_provider}\n")
        f.write(f"- **Generator Model:** {args.generator_model}\n")
        f.write(f"- **Reflector Model:** {args.reflector_model}\n")
        f.write(f"- **Curator Model:** {args.curator_model}\n")
        f.write(f"- **Training Samples:** {n_train}\n")
        f.write(f"- **Validation Samples:** {n_val}\n")
        f.write(f"- **Test Samples:** {n_test}\n\n")
        
        # Results Summary
        f.write("## Results Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        if baseline is not None:
            f.write(f"| Baseline Accuracy (no playbook) | {baseline:.4f} ({baseline*100:.2f}%) |\n")
        else:
            f.write("| Baseline Accuracy (no playbook) | N/A (skipped) |\n")
        
        if final is not None:
            f.write(f"| ACE Accuracy (with playbook) | {final:.4f} ({final*100:.2f}%) |\n")
        else:
            f.write("| ACE Accuracy (with playbook) | N/A |\n")
        
        if improvement is not None:
            sign = "+" if improvement >= 0 else ""
            f.write(f"| **Improvement** | **{sign}{improvement:.4f} ({sign}{improvement*100:.2f}%)** |\n")
        
        f.write("\n")
        
        # Playbook
        f.write("## Best Playbook\n\n")
        if best_playbook:
            f.write("The following playbook was learned by ACE during training:\n\n")
            f.write("```\n")
            f.write(best_playbook)
            f.write("\n```\n\n")
        else:
            f.write("No playbook was generated.\n\n")
        
        # Task Description
        f.write("## Task Description\n\n")
        f.write("**Dataset:** MBPP (Mostly Basic Python Problems)\n\n")
        f.write("**Input:** Natural language problem description + test cases\n\n")
        f.write("**Output:** Python code solution\n\n")
        f.write("**Evaluation:** Code execution - all test cases must pass\n\n")
        
        # Files
        f.write("## Output Files\n\n")
        f.write(f"- **Report:** `REPORT.md`\n")
        f.write(f"- **Summary JSON:** `summary.json`\n")
        if results_summary.get('playbook_path'):
            f.write(f"- **Best Playbook:** `best_playbook.txt`\n")
        f.write(f"- **Results Directory:** `{results_dir}`\n")
    
    return report_path


def check_api_key():
    """Verify API key is set."""
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)
    return api_key


def load_coding_data(max_train=None, max_test=None):
    """Load and process coding dataset (MBPP)."""
    from coding.data_processor import DataProcessor, load_data
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data")
    
    # Load raw data
    train_raw = load_data(os.path.join(data_path, "train.csv"))
    val_raw = load_data(os.path.join(data_path, "val.csv"))
    test_raw = load_data(os.path.join(data_path, "test.csv"))
    
    # Limit samples if specified
    if max_train:
        train_raw = train_raw[:max_train]
        val_raw = val_raw[:max(max_train // 4, 10)]
    if max_test:
        test_raw = test_raw[:max_test]
    
    # Process data
    processor = DataProcessor(task_name="coding")
    train_samples = processor.process_task_data(train_raw)
    val_samples = processor.process_task_data(val_raw)
    test_samples = processor.process_task_data(test_raw)
    
    return train_samples, val_samples, test_samples, processor


def run_ace_training(ace_system, train_samples, val_samples, test_samples, processor, results_dir):
    """Train ACE to improve the playbook (includes initial and final evaluations)."""
    log("\n" + "="*70)
    log("ACE TRAINING (Offline Mode)")
    log("="*70)
    log(f"Training on {len(train_samples)} samples, validating on {len(val_samples)}")
    
    config = {
        'num_epochs': 1,
        'max_num_rounds': 3,  # Max reflection rounds per sample
        'curator_frequency': 5,  # Run curator every step
        'eval_steps': max(len(train_samples) // 10, 10),  # Evaluate 10 times during training
        'save_steps': max(len(train_samples) // 10, 10),
        'playbook_token_budget': 80000,
        'task_name': 'coding_ace',
        'json_mode': False,
        'no_ground_truth': False,
        'save_dir': os.path.join(results_dir, "training"),
        'test_workers': 10,
    }
    
    log(f"\nTraining configuration:")
    log(f"  - Epochs: {config['num_epochs']}")
    log(f"  - Max reflection rounds: {config['max_num_rounds']}")
    log(f"  - Curator frequency: every {config['curator_frequency']} steps")
    log(f"  - Evaluation steps: {config['eval_steps']}")
    
    results = ace_system.run(
        mode='offline',
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=processor,
        config=config
    )
    
    # Extract results
    initial_acc = results.get('initial_test_results', {}).get('accuracy', 0)
    final_acc = results.get('final_test_results', {}).get('accuracy', 0)
    training_results = results.get('training_results', {})
    
    log(f"\n📊 Training Results:")
    log(f"  - Initial test accuracy: {initial_acc:.4f}")
    log(f"  - Best validation accuracy: {training_results.get('best_val_accuracy', 'N/A')}")
    log(f"  - Final test accuracy: {final_acc:.4f}")
    
    # Save the best playbook
    playbook_path = os.path.join(results_dir, "best_playbook.txt")
    with open(playbook_path, 'w') as f:
        f.write(ace_system.best_playbook)
    log(f"\n💾 Best playbook saved to: {playbook_path}")
    
    return ace_system.best_playbook, results


def main():
    parser = argparse.ArgumentParser(description='ACE Coding Workflow (MBPP)')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Limit training samples (for faster testing)')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Limit test samples (for faster testing)')
    parser.add_argument('--api-provider', type=str, default='anthropic',
                        choices=['anthropic', 'openai', 'together', 'sambanova'],
                        help='API provider to use')
    parser.add_argument('--generator-model', type=str, default='claude-haiku-4-5',
                        help='Model to use for generator agent')
    parser.add_argument('--reflector-model', type=str, default='claude-sonnet-4-5',
                        help='Model to use for reflector agent')
    parser.add_argument('--curator-model', type=str, default='claude-sonnet-4-5',
                        help='Model to use for curator agent')
    args = parser.parse_args()
    
    log("\n" + "#"*70)
    log("#" + " "*12 + "ACE PYTHON CODE GENERATION WORKFLOW" + " "*15 + "#")
    log("#"*70)
    
    # Check API key
    check_api_key()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results", f"coding_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    log(f"\n📁 Results will be saved to: {results_dir}")
    
    # Load data
    log("\n📦 Loading MBPP coding dataset...")
    train_samples, val_samples, test_samples, processor = load_coding_data(
        max_train=args.max_train,
        max_test=args.max_test
    )
    log(f"  - Train: {len(train_samples)} samples")
    log(f"  - Val: {len(val_samples)} samples")
    log(f"  - Test: {len(test_samples)} samples")
    
    # Coding-specific playbook template with domain-relevant sections
    CODING_PLAYBOOK_TEMPLATE = """## GENERAL

## CODE GENERATION PRINCIPLES

## COMMON PYTHON PATTERNS

## HANDLING EDGE CASES

## ALGORITHM DESIGN

## TEST CASE INTERPRETATION

## COMMON MISTAKES TO AVOID

## DEBUGGING STRATEGIES

## OTHERS"""
    
    # Initialize ACE
    from ace import ACE
    
    log(f"\n🤖 Initializing ACE system...")
    log(f"  - Provider: {args.api_provider}")
    log(f"  - Generator model: {args.generator_model}")
    log(f"  - Reflector model: {args.reflector_model}")
    log(f"  - Curator model: {args.curator_model}")
    
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=4096,
        initial_playbook=CODING_PLAYBOOK_TEMPLATE,
        use_bulletpoint_analyzer=True,
        bulletpoint_analyzer_threshold=0.9,
        generator_temperature=0.1,
        reflector_temperature=0.7,
        curator_temperature=0.7
    )
    
    # Results tracking
    results_summary = {
        'baseline_accuracy': None,
        'final_accuracy': None,
        'improvement': None,
        'playbook_path': None
    }

    # ACE Training (includes initial and final evaluations)
    best_playbook, training_results = run_ace_training(
        ace_system, train_samples, val_samples, test_samples, processor, results_dir
    )
    results_summary['playbook_path'] = os.path.join(results_dir, "best_playbook.txt")
    
    # Extract accuracies from training results
    results_summary['baseline_accuracy'] = training_results.get('initial_test_results', {}).get('accuracy')
    results_summary['final_accuracy'] = training_results.get('final_test_results', {}).get('accuracy')
    
    # Calculate improvement
    if results_summary['baseline_accuracy'] and results_summary['final_accuracy']:
        improvement = results_summary['final_accuracy'] - results_summary['baseline_accuracy']
        results_summary['improvement'] = improvement
    
    # Final Summary
    log("\n" + "="*70)
    log("📊 FINAL SUMMARY")
    log("="*70)
    
    if results_summary['baseline_accuracy']:
        log(f"  Baseline Accuracy:  {results_summary['baseline_accuracy']:.4f} ({results_summary['baseline_accuracy']*100:.2f}%)")
    
    if results_summary['final_accuracy']:
        log(f"  Final Accuracy:     {results_summary['final_accuracy']:.4f} ({results_summary['final_accuracy']*100:.2f}%)")
    
    if results_summary['improvement'] is not None:
        sign = "+" if results_summary['improvement'] >= 0 else ""
        log(f"  Improvement:        {sign}{results_summary['improvement']:.4f} ({sign}{results_summary['improvement']*100:.2f}%)")
    
    if results_summary['playbook_path']:
        log(f"\n  📖 Best Playbook: {results_summary['playbook_path']}")
    
    log(f"  📁 All Results: {results_dir}")
    
    # Save JSON summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    log(f"  📄 Summary JSON: {summary_path}")
    
    # Save human-readable report
    report_path = os.path.join(results_dir, "REPORT.md")
    save_report(results_dir, results_summary, best_playbook, args, 
                len(train_samples), len(val_samples), len(test_samples))
    log(f"  📄 Report: {report_path}")
    
    log("\n" + "="*70)
    log("✅ Workflow completed!")
    log("="*70 + "\n")
    
    return results_summary


if __name__ == "__main__":
    main()
