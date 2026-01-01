#!/usr/bin/env python3
"""
Complete GEPA workflow for Python Code Generation (MBPP dataset).

This script runs the full GEPA workflow:
1. Evaluate baseline (ChainOfThought without optimization) on test set
2. Train GEPA to optimize the prompt
3. Evaluate with the optimized prompt from GEPA

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    cd /path/to/prompt_optimisation_gepa_ace
    source .venv/bin/activate
    cd ace
    python -m coding.run_gepa_workflow

Options:
    --skip-baseline     Skip initial baseline evaluation
    --skip-training     Skip GEPA training
    --max-train N       Limit training samples (for faster testing)
    --max-test N        Limit test samples (for faster testing)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Any

import dspy
import pandas as pd

# Load config
with open('/Users/marie/Documents/github/config.json', 'r') as f:
    config = json.load(f)
    for key in config:
        os.environ[key] = config[key]

# Global log file handle
_log_file = None

def set_log_file(filepath: str):
    """Set the file path for logging output."""
    global _log_file
    if _log_file:
        _log_file.close()
    _log_file = open(filepath, 'a', encoding='utf-8')

def log(message: str = "", end: str = "\n"):
    """Print with timestamp and GEPA prefix, and save to file if configured."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[GEPA {timestamp}] {message}"
    print(formatted, end=end, flush=True)
    if _log_file:
        _log_file.write(formatted + end)
        _log_file.flush()


class CodeResponse(dspy.Signature):
    """You are an expert Python programmer. Given a problem description and test cases, write a Python function that solves the problem and passes all tests. Focus on correctness, edge cases, and clean code."""
    
    problem = dspy.InputField(desc='Problem description in natural language')
    test_cases = dspy.InputField(desc='Test assertions the code must pass')
    code: str = dspy.OutputField(desc='Python code solution - function only, no explanations')


def metric(gold, pred, trace=None):
    """
    Simple execution-based metric for code generation.
    
    Executes the predicted code with the test cases from the gold example.
    """
    from coding.data_processor import extract_code_from_response, execute_code_with_tests, parse_test_list
    
    # Extract test cases from gold example - gold is a dspy.Example object
    test_cases_str = gold.test_cases if hasattr(gold, 'test_cases') else gold.get('test_cases', '')
    test_list = parse_test_list(test_cases_str)
    
    if not test_list:
        # Debug: log when no test list found
        print(f"[METRIC DEBUG] No test list found. gold type: {type(gold)}, test_cases_str: {test_cases_str[:100] if test_cases_str else 'None'}")
        return 0.0
    
    # Extract code from prediction
    try:
        predicted_code = pred.code
    except AttributeError:
        # Debug: log when no code attribute
        print(f"[METRIC DEBUG] No code attribute in prediction. pred type: {type(pred)}, pred: {pred}")
        return 0.0
    
    if not predicted_code or not predicted_code.strip():
        print(f"[METRIC DEBUG] Empty predicted code")
        return 0.0
    
    code = extract_code_from_response(predicted_code)
    
    if not code or not code.strip():
        print(f"[METRIC DEBUG] No code extracted from response")
        return 0.0
    
    # Execute code with tests
    result = execute_code_with_tests(code, test_list, timeout=5)
    
    # Return 1.0 for success, 0.0 for failure (must be float for DSPy)
    return 1.0 if result['success'] else 0.0

# Initialize failure counter
metric._fail_count = 0


def metric_with_feedback(
    example: dspy.Example, 
    prediction: dspy.Prediction, 
    trace=None, 
    pred_name=None, 
    pred_trace=None
) -> dspy.Prediction:
    """
    Enhanced evaluation metric with detailed feedback for GEPA optimization.
    
    Evaluates code generation predictions and generates targeted feedback 
    to help GEPA identify failure patterns and improve prompts for code generation.
    """
    from coding.data_processor import extract_code_from_response, execute_code_with_tests, parse_test_list
    
    # Extract problem and test cases - handle both dict and dspy.Example
    problem = example.problem if hasattr(example, 'problem') else example.get('problem', '')
    test_cases_str = example.test_cases if hasattr(example, 'test_cases') else example.get('test_cases', '')
    test_list = parse_test_list(test_cases_str)
    
    if not test_list:
        print(f"[METRIC_WITH_FEEDBACK DEBUG] No test list found. test_cases_str: {test_cases_str[:100] if test_cases_str else 'None'}")
        feedback_text = (
            f"No test cases found to validate the code.\n\n"
            f"Problem: {problem}"
        )
        return dspy.Prediction(score=0.0, feedback=feedback_text)
    
    # Extract predicted code
    try:
        predicted_code = prediction.code
    except AttributeError:
        print(f"[METRIC_WITH_FEEDBACK DEBUG] No code attribute in prediction")
        feedback_text = (
            f"The prediction must include a 'code' field with Python code. "
            f"Your response didn't contain valid code. "
            f"Please ensure you generate a complete Python function.\n\n"
            f"Problem: {problem}\n"
            f"Test cases:\n{test_cases_str}"
        )
        return dspy.Prediction(score=0.0, feedback=feedback_text)
    
    if not predicted_code or not predicted_code.strip():
        print(f"[METRIC_WITH_FEEDBACK DEBUG] Empty predicted code")
        feedback_text = (
            f"Empty code provided. You must generate Python code to solve the problem.\n\n"
            f"Problem: {problem}\n"
            f"Test cases:\n{test_cases_str}"
        )
        return dspy.Prediction(score=0.0, feedback=feedback_text)
    
    # Extract and execute code
    code = extract_code_from_response(predicted_code)
    
    if not code or not code.strip():
        print(f"[METRIC_WITH_FEEDBACK DEBUG] No code extracted from response")
        feedback_text = (
            f"Could not extract valid Python code from your response.\n\n"
            f"Problem: {problem}\n"
            f"Test cases:\n{test_cases_str}"
        )
        return dspy.Prediction(score=0.0, feedback=feedback_text)
    
    result = execute_code_with_tests(code, test_list, timeout=5)
    
    # Score: 1.0 if all tests pass, 0.0 otherwise
    score = 1.0 if result['success'] else 0.0
    
    # Debug: Print score for first few examples
    if not hasattr(metric_with_feedback, '_call_count'):
        metric_with_feedback._call_count = 0
    metric_with_feedback._call_count += 1
    if metric_with_feedback._call_count <= 10 or (metric_with_feedback._call_count % 50 == 0):
        print(f"[METRIC_WITH_FEEDBACK DEBUG] Call {metric_with_feedback._call_count}: score={score}, passed={result['passed']}/{result['total']}, problem={problem[:50]}...")
        if metric_with_feedback._call_count <= 3:
            print(f"  Code: {code[:100]}...")
            if result['errors']:
                print(f"  Error: {result['errors'][0][:150]}...")
    
    # Generate detailed feedback based on execution results
    if score == 1:
        feedback_text = (
            f"✓ Correct! Your code passes all {result['total']} test cases.\n\n"
            f"Problem: {problem}\n"
            f"Your code:\n{code[:200]}..." if len(code) > 200 else f"Your code:\n{code}"
        )
    else:
        # Analyze failure type
        if result['timeout']:
            feedback_text = (
                f"✗ Code execution timed out.\n\n"
                f"Problem: {problem}\n"
                f"Your code:\n{code[:200]}...\n\n" if len(code) > 200 else f"Your code:\n{code}\n\n"
                f"ERROR: Your code took too long to execute (>5 seconds). "
                f"This suggests:\n"
                f"1. An infinite loop\n"
                f"2. Inefficient algorithm with high time complexity\n"
                f"3. Unnecessary recursive calls\n\n"
                f"Review your algorithm and ensure it terminates quickly for all inputs."
            )
        elif any('Syntax error' in err for err in result['errors']):
            feedback_text = (
                f"✗ Syntax error in generated code.\n\n"
                f"Problem: {problem}\n"
                f"Your code:\n{code}\n\n"
                f"ERRORS:\n" + "\n".join(result['errors']) + "\n\n"
                f"Fix the syntax errors. Ensure:\n"
                f"1. Proper indentation\n"
                f"2. Correct use of colons, parentheses, brackets\n"
                f"3. Valid Python syntax\n"
                f"4. Proper function definition"
            )
        elif result['passed'] == 0:
            feedback_text = (
                f"✗ All test cases failed ({result['passed']}/{result['total']} passed).\n\n"
                f"Problem: {problem}\n"
                f"Your code:\n{code}\n\n"
                f"ERRORS:\n" + "\n".join(result['errors'][:3]) + "\n\n"  # Show first 3 errors
                f"Your code has fundamental issues. Check:\n"
                f"1. Function name matches what tests expect\n"
                f"2. Function signature (parameters) is correct\n"
                f"3. Return value type and format\n"
                f"4. Logic actually solves the problem\n"
                f"5. You're not missing any edge cases"
            )
        else:
            feedback_text = (
                f"✗ Partial success ({result['passed']}/{result['total']} tests passed).\n\n"
                f"Problem: {problem}\n"
                f"Your code:\n{code}\n\n"
                f"ERRORS:\n" + "\n".join(result['errors']) + "\n\n"
                f"Your code works for some cases but fails others. This suggests:\n"
                f"1. Missing edge case handling (empty input, None, negative numbers, etc.)\n"
                f"2. Incorrect logic for specific conditions\n"
                f"3. Off-by-one errors in loops or indices\n"
                f"4. Type conversion issues\n\n"
                f"Analyze the failing test cases carefully and handle all edge cases."
            )
    
    # For GEPA, we need to return a Prediction with score AND set it as return value
    # The score attribute is for feedback display, the return value is for DSPy evaluation
    result_pred = dspy.Prediction(score=float(score), feedback=feedback_text)
    # Also make it so float(result_pred) returns the score
    result_pred._score = float(score)
    return result_pred


def save_report(results_dir, results_summary, optimized_instructions, args, n_train, n_val, n_test):
    """Save a human-readable markdown report with all results."""
    
    report_path = os.path.join(results_dir, "REPORT.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate improvement
    baseline = results_summary.get('baseline_accuracy')
    final = results_summary.get('final_accuracy')
    improvement = results_summary.get('improvement')
    
    with open(report_path, 'w') as f:
        f.write("# GEPA Python Code Generation (MBPP) - Run Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        
        # Configuration
        f.write("## Configuration\n\n")
        f.write(f"- **Main LM:** {args.main_model}\n")
        f.write(f"- **Reflection LM:** {args.reflection_model}\n")
        f.write(f"- **Training Samples:** {n_train}\n")
        f.write(f"- **Validation Samples:** {n_val}\n")
        f.write(f"- **Test Samples:** {n_test}\n")
        f.write(f"- **Num Threads:** {args.num_threads}\n\n")
        
        # Results Summary
        f.write("## Results Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        if baseline is not None:
            f.write(f"| Baseline Accuracy (no optimization) | {baseline:.4f} ({baseline*100:.2f}%) |\n")
        else:
            f.write("| Baseline Accuracy (no optimization) | N/A (skipped) |\n")
        
        if final is not None:
            f.write(f"| GEPA Accuracy (optimized) | {final:.4f} ({final*100:.2f}%) |\n")
        else:
            f.write("| GEPA Accuracy (optimized) | N/A |\n")
        
        if improvement is not None:
            sign = "+" if improvement >= 0 else ""
            f.write(f"| **Improvement** | **{sign}{improvement:.4f} ({sign}{improvement*100:.2f}%)** |\n")
        
        f.write("\n")
        
        # Task Description
        f.write("## Task Description\n\n")
        f.write("**Dataset:** MBPP (Mostly Basic Python Problems)\n\n")
        f.write("**Input:** Natural language problem description + test cases\n\n")
        f.write("**Output:** Python code solution\n\n")
        f.write("**Evaluation:** Code execution - all test cases must pass\n\n")
        
        # Optimized Instructions
        f.write("## Optimized Instructions\n\n")
        if optimized_instructions:
            f.write("The following instructions were learned by GEPA during training:\n\n")
            f.write("```\n")
            f.write(optimized_instructions)
            f.write("\n```\n\n")
        else:
            f.write("No optimized instructions were generated.\n\n")
        
        # Files
        f.write("## Output Files\n\n")
        f.write(f"- **Report:** `REPORT.md`\n")
        f.write(f"- **Summary JSON:** `summary.json`\n")
        if results_summary.get('instructions_path'):
            f.write(f"- **Optimized Instructions:** `optimized_instructions.txt`\n")
        f.write(f"- **Results Directory:** `{results_dir}`\n")
    
    return report_path


def load_coding_data(max_train=None, max_test=None):
    """Load and process coding dataset into DSPy Examples."""
    from coding.data_processor import parse_test_list
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data")
    
    # Load raw data
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    
    # Limit samples if specified
    if max_train:
        train_df = train_df.head(max_train)
        val_df = val_df.head(max(max_train // 4, 10))
    if max_test:
        test_df = test_df.head(max_test)
    
    # Convert to DSPy Examples
    def create_example(row):
        test_list = parse_test_list(row['test_list'])
        test_cases_formatted = '\n'.join(test_list)
        
        return dspy.Example({
            "problem": row['text'],
            "test_cases": row['test_list'],  # Keep original for parsing
            "code": row['code'],
        }).with_inputs("problem", "test_cases")
    
    train_set = [create_example(row) for row in train_df.to_dict(orient='records')]
    val_set = [create_example(row) for row in val_df.to_dict(orient='records')]
    test_set = [create_example(row) for row in test_df.to_dict(orient='records')]
    
    return train_set, val_set, test_set


def run_baseline_evaluation(test_set, num_threads, results_dir):
    """Step 1: Evaluate baseline (ChainOfThought without optimization) on test set."""
    log("\n" + "="*70)
    log("STEP 1: BASELINE EVALUATION (No Optimization)")
    log("="*70)
    
    # Create baseline program
    baseline_program = dspy.ChainOfThought(CodeResponse)
    
    # Create evaluator
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=metric,
        num_threads=num_threads,
        display_table=False,
        display_progress=True
    )
    
    # Run evaluation
    log(f"\nEvaluating on {len(test_set)} test samples...")
    eval_result = evaluate(baseline_program)
    
    # Extract accuracy from EvaluationResult object
    accuracy = float(eval_result)
    
    log(f"\n📊 Baseline Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save baseline results
    baseline_dir = os.path.join(results_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    
    baseline_results = {
        'accuracy': accuracy,
        'num_samples': len(test_set),
        'instructions': baseline_program.predict.signature.instructions
    }
    
    with open(os.path.join(baseline_dir, "results.json"), 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    log(f"  Results saved to: baseline/results.json")
    
    return accuracy, baseline_program


def run_gepa_training(train_set, val_set, main_lm, reflection_lm, num_threads, results_dir, auto_mode="medium"):
    """Step 2: Train GEPA to optimize the prompt."""
    log("\n" + "="*70)
    log("STEP 2: GEPA TRAINING")
    log("="*70)
    log(f"Training on {len(train_set)} samples, validating on {len(val_set)}")
    
    from dspy import GEPA
    
    # Create a fresh program for optimization
    program_to_optimize = dspy.ChainOfThought(CodeResponse)
    
    # Initialize GEPA optimizer
    gepa_optimizer = GEPA(
        metric=metric_with_feedback,
        auto=auto_mode,
        num_threads=num_threads,
        track_stats=True,
        reflection_minibatch_size=32,
        track_best_outputs=True,
        add_format_failure_as_feedback=True,
        reflection_lm=reflection_lm,
    )
    
    log(f"\nGEPA configuration:")
    log(f"  - Auto mode: {auto_mode}")
    log(f"  - Num threads: {num_threads}")
    log(f"  - Reflection minibatch size: 32")
    
    # Run optimization
    log(f"\nRunning GEPA optimization...")
    optimized_program = gepa_optimizer.compile(
        program_to_optimize,
        trainset=train_set,
        valset=val_set,
    )
    
    # Extract optimized instructions
    optimized_instructions = optimized_program.predict.signature.instructions
    
    log(f"\n📝 Optimized Instructions:")
    log(f"  {optimized_instructions[:200]}..." if len(optimized_instructions) > 200 else f"  {optimized_instructions}")
    
    # Save training results
    training_dir = os.path.join(results_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    
    # Save optimized instructions
    instructions_path = os.path.join(results_dir, "optimized_instructions.txt")
    with open(instructions_path, 'w') as f:
        f.write(optimized_instructions)
    log(f"\n💾 Optimized instructions saved to: {instructions_path}")
    
    # Save training metadata
    training_results = {
        'optimized_instructions': optimized_instructions,
        'num_train_samples': len(train_set),
        'num_val_samples': len(val_set),
    }
    
    # Check if detailed results are available
    if hasattr(optimized_program, 'detailed_results'):
        detailed = optimized_program.detailed_results
        training_results['best_idx'] = detailed.best_idx
        training_results['val_aggregate_scores'] = detailed.val_aggregate_scores
        log(f"  - Best candidate index: {detailed.best_idx}")
        log(f"  - Validation scores: {detailed.val_aggregate_scores}")
    
    with open(os.path.join(training_dir, "results.json"), 'w') as f:
        json.dump(training_results, f, indent=2)
    
    return optimized_program, optimized_instructions


def run_final_evaluation(optimized_program, test_set, num_threads, results_dir):
    """Step 3: Evaluate with the optimized prompt from GEPA."""
    log("\n" + "="*70)
    log("STEP 3: FINAL EVALUATION (With Optimized Prompt)")
    log("="*70)
    
    # Create evaluator
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=metric,
        num_threads=num_threads,
        display_table=False,
        display_progress=True
    )
    
    # Run evaluation
    log(f"\nEvaluating on {len(test_set)} test samples...")
    eval_result = evaluate(optimized_program)
    
    # Extract accuracy from EvaluationResult object
    accuracy = float(eval_result)
    
    log(f"\n🎯 Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save final results
    final_dir = os.path.join(results_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    final_results = {
        'accuracy': accuracy,
        'num_samples': len(test_set),
        'instructions': optimized_program.predict.signature.instructions
    }
    
    with open(os.path.join(final_dir, "results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='GEPA Coding Workflow (MBPP)')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip initial baseline evaluation')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip GEPA training')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Limit training samples (for faster testing)')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Limit test samples (for faster testing)')
    parser.add_argument('--main-model', type=str, default='anthropic/claude-haiku-4-5',
                        help='Model to use for main LM')
    parser.add_argument('--reflection-model', type=str, default='anthropic/claude-sonnet-4-20250514',
                        help='Model to use for reflection LM')
    parser.add_argument('--num-threads', type=int, default=16,
                        help='Number of threads for parallel evaluation')
    parser.add_argument('--auto-mode', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='GEPA auto mode: light (fast), medium (balanced), heavy (thorough)')
    args = parser.parse_args()
    
    log("\n" + "#"*70)
    log("#" + " "*10 + "GEPA PYTHON CODE GENERATION WORKFLOW" + " "*16 + "#")
    log("#"*70)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results", f"gepa_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    log(f"\n📁 Results will be saved to: {results_dir}")
    
    # Set up log file
    log_file_path = os.path.join(results_dir, "detailed_log.txt")
    set_log_file(log_file_path)
    log(f"📝 Detailed logs will be saved to: {log_file_path}")
    
    # Initialize LMs
    log(f"\n🤖 Initializing DSPy LMs...")
    log(f"  - Main model: {args.main_model}")
    log(f"  - Reflection model: {args.reflection_model}")
    
    main_lm = dspy.LM(
        args.main_model,
        temperature=0.1
    )
    
    reflection_lm = dspy.LM(
        args.reflection_model,
        temperature=0.4
    )
    
    dspy.configure(lm=main_lm)
    
    # Load data
    log("\n📦 Loading MBPP coding dataset...")
    train_set, val_set, test_set = load_coding_data(
        max_train=args.max_train,
        max_test=args.max_test
    )
    log(f"  - Train: {len(train_set)} samples")
    log(f"  - Val: {len(val_set)} samples")
    log(f"  - Test: {len(test_set)} samples")
    
    # Results tracking
    results_summary = {
        'baseline_accuracy': None,
        'final_accuracy': None,
        'improvement': None,
        'instructions_path': None
    }
    
    optimized_instructions = ""
    optimized_program = None
    
    # Step 1: Baseline evaluation
    if not args.skip_baseline:
        baseline_acc, _ = run_baseline_evaluation(
            test_set, args.num_threads, results_dir
        )
        results_summary['baseline_accuracy'] = baseline_acc
    else:
        log("\n⏭️  Skipping baseline evaluation")
    
    # Step 2: GEPA Training
    if not args.skip_training:
        optimized_program, optimized_instructions = run_gepa_training(
            train_set, val_set, main_lm, reflection_lm, args.num_threads, results_dir,
            auto_mode=args.auto_mode
        )
        results_summary['instructions_path'] = os.path.join(results_dir, "optimized_instructions.txt")
    else:
        log("\n⏭️  Skipping GEPA training")
    
    # Step 3: Final evaluation with optimized program
    if optimized_program is not None:
        final_acc = run_final_evaluation(
            optimized_program, test_set, args.num_threads, results_dir
        )
        results_summary['final_accuracy'] = final_acc
    
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
    
    if results_summary['instructions_path']:
        log(f"\n  📝 Optimized Instructions: {results_summary['instructions_path']}")
    
    log(f"  📁 All Results: {results_dir}")
    
    # Save JSON summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    log(f"  📄 Summary JSON: {summary_path}")
    
    # Save human-readable report
    report_path = save_report(results_dir, results_summary, optimized_instructions, args, 
                              len(train_set), len(val_set), len(test_set))
    log(f"  📄 Report: {report_path}")
    
    log("\n" + "="*70)
    log("✅ Workflow completed!")
    log("="*70 + "\n")
    
    return results_summary


if __name__ == "__main__":
    main()
