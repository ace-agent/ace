#!/usr/bin/env python3
"""
Complete GEPA workflow for Banking Topic Classification.

This script runs the full GEPA workflow:
1. Evaluate baseline (ChainOfThought without optimization) on test set
2. Train GEPA to optimize the prompt
3. Evaluate with the optimized prompt from GEPA

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    cd /path/to/ace
    python3 -m banking.run_gepa_workflow

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
from typing import Literal

import dspy
import pandas as pd

# Load config
with open('/Users/marie/Documents/github/config.json', 'r') as f:
    config = json.load(f)
    for key in config:
        os.environ[key] = config[key]


def log(message: str = "", end: str = "\n"):
    """Print with timestamp and GEPA prefix for easy log identification."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[GEPA {timestamp}] {message}", end=end, flush=True)


# Define allowed banking topics
ALLOWED_TOPICS = [
    'card_arrival', 'card_linking', 'exchange_rate', 'card_payment_wrong_exchange_rate', 
    'extra_charge_on_statement', 'pending_cash_withdrawal', 'fiat_currency_support', 
    'card_delivery_estimate', 'automatic_top_up', 'card_not_working', 'exchange_via_app', 
    'lost_or_stolen_card', 'age_limit', 'pin_blocked', 'contactless_not_working', 
    'top_up_by_bank_transfer_charge', 'pending_top_up', 'cancel_transfer', 'top_up_limits', 
    'wrong_amount_of_cash_received', 'card_payment_fee_charged', 'transfer_not_received_by_recipient', 
    'supported_cards_and_currencies', 'getting_virtual_card', 'card_acceptance', 'top_up_reverted', 
    'balance_not_updated_after_cheque_or_cash_deposit', 'card_payment_not_recognised', 
    'edit_personal_details', 'why_verify_identity', 'unable_to_verify_identity', 'get_physical_card', 
    'visa_or_mastercard', 'topping_up_by_card', 'disposable_card_limits', 'compromised_card', 
    'atm_support', 'direct_debit_payment_not_recognised', 'passcode_forgotten', 
    'declined_cash_withdrawal', 'pending_card_payment', 'lost_or_stolen_phone', 'request_refund', 
    'declined_transfer', 'Refund_not_showing_up', 'declined_card_payment', 'pending_transfer', 
    'terminate_account', 'card_swallowed', 'transaction_charged_twice', 'verify_source_of_funds', 
    'transfer_timing', 'reverted_card_payment?', 'change_pin', 'beneficiary_not_allowed', 
    'transfer_fee_charged', 'receiving_money', 'failed_transfer', 'transfer_into_account', 
    'verify_top_up', 'getting_spare_card', 'top_up_by_cash_or_cheque', 'order_physical_card', 
    'virtual_card_not_working', 'wrong_exchange_rate_for_cash_withdrawal', 'get_disposable_virtual_card', 
    'top_up_failed', 'balance_not_updated_after_bank_transfer', 'cash_withdrawal_not_recognised', 
    'exchange_charge', 'top_up_by_card_charge', 'activate_my_card', 'cash_withdrawal_charge', 
    'card_about_to_expire', 'apple_pay_or_google_pay', 'verify_my_identity', 'country_support'
]


class TopicResponse(dspy.Signature):
    """Classify the problem into one of the predefined categories."""
    question = dspy.InputField(desc='Customer question')
    topic: Literal['card_arrival', 'card_linking', 'exchange_rate', 'card_payment_wrong_exchange_rate', 
                   'extra_charge_on_statement', 'pending_cash_withdrawal', 
                   'fiat_currency_support', 'card_delivery_estimate', 
                   'automatic_top_up', 'card_not_working', 'exchange_via_app', 
                   'lost_or_stolen_card', 'age_limit', 'pin_blocked', 
                   'contactless_not_working', 'top_up_by_bank_transfer_charge', 
                   'pending_top_up', 'cancel_transfer', 'top_up_limits', 
                   'wrong_amount_of_cash_received', 'card_payment_fee_charged', 
                   'transfer_not_received_by_recipient', 'supported_cards_and_currencies', 
                   'getting_virtual_card', 'card_acceptance', 'top_up_reverted', 
                   'balance_not_updated_after_cheque_or_cash_deposit', 'card_payment_not_recognised', 
                   'edit_personal_details', 'why_verify_identity', 'unable_to_verify_identity', 
                   'get_physical_card', 'visa_or_mastercard', 'topping_up_by_card', 'disposable_card_limits', 
                   'compromised_card', 'atm_support', 'direct_debit_payment_not_recognised', 
                   'passcode_forgotten', 'declined_cash_withdrawal', 'pending_card_payment', 
                   'lost_or_stolen_phone', 'request_refund', 'declined_transfer', 
                   'Refund_not_showing_up', 'declined_card_payment', 'pending_transfer', 
                   'terminate_account', 'card_swallowed', 'transaction_charged_twice', 
                   'verify_source_of_funds', 'transfer_timing', 'reverted_card_payment?', 
                   'change_pin', 'beneficiary_not_allowed', 'transfer_fee_charged', 
                   'receiving_money', 'failed_transfer', 'transfer_into_account', 'verify_top_up', 
                   'getting_spare_card', 'top_up_by_cash_or_cheque', 'order_physical_card', 
                   'virtual_card_not_working', 'wrong_exchange_rate_for_cash_withdrawal', 
                   'get_disposable_virtual_card', 'top_up_failed', 
                   'balance_not_updated_after_bank_transfer', 'cash_withdrawal_not_recognised', 
                   'exchange_charge', 'top_up_by_card_charge', 'activate_my_card', 
                   'cash_withdrawal_charge', 'card_about_to_expire', 'apple_pay_or_google_pay', 
                   'verify_my_identity', 'country_support'] = dspy.OutputField(desc='Topic from the predefined categories')


def metric(gold, pred, trace=None):
    """Simple accuracy metric for topic classification."""
    return gold.topic.lower() == pred.topic.lower()


def metric_with_feedback(
    example: dspy.Example, 
    prediction: dspy.Prediction, 
    trace=None, 
    pred_name=None, 
    pred_trace=None
) -> dspy.Prediction:
    """
    Enhanced evaluation metric with detailed feedback for GEPA optimization.
    
    Evaluates topic classification predictions and generates targeted feedback 
    to help GEPA identify failure patterns and improve prompts for categorization tasks.
    """
    # Extract ground truth topic and the question
    correct_topic = example.get('topic', '')
    question = example.get('question', '')
    
    # Extract predicted topic
    try:
        predicted_topic = prediction.topic
    except AttributeError:
        # Handle case where topic attribute is missing
        feedback_text = (
            f"The prediction must include a 'topic' field from the predefined categories. "
            f"Your response didn't contain a valid topic classification. "
            f"Please ensure you select exactly one category from the allowed options."
        )
        feedback_text += f"\n\nFor the question: '{question}'\nThe correct topic is: '{correct_topic}'."
        feedback_text += (
            f"\n\nAnalyze the keywords and intent in this question to understand "
            f"why '{correct_topic}' is the appropriate category."
        )
        return dspy.Prediction(score=0, feedback=feedback_text)
    
    # Validate that predicted topic is in allowed list
    if predicted_topic not in ALLOWED_TOPICS:
        feedback_text = (
            f"⚠ Invalid topic classification.\n\n"
            f"Question: '{question}'\n"
            f"Your classification: '{predicted_topic}'\n"
            f"Correct classification: '{correct_topic}'\n\n"
            f"ERROR: The topic '{predicted_topic}' is NOT in the list of allowed categories. "
            f"You must select EXACTLY one topic from the predefined list of 77 banking categories. "
            f"Do not create new categories or use variations of existing ones. "
            f"Review the allowed categories and ensure your response uses the exact category name "
            f"as specified (including underscores, capitalization, etc.)."
        )
        return dspy.Prediction(score=0, feedback=feedback_text)

    # Score: 1 for correct, 0 for incorrect
    score = metric(example, prediction)

    # Generate appropriate feedback based on correctness
    if score == 1:
        feedback_text = (
            f"✓ Correct! You classified the question correctly as '{correct_topic}'.\n\n"
            f"Question: '{question}'\n"
            f"Your classification: '{predicted_topic}'"
        )
    else:
        feedback_text = (
            f"✗ Incorrect classification.\n\n"
            f"Question: '{question}'\n"
            f"Your classification: '{predicted_topic}'\n"
            f"Correct classification: '{correct_topic}'\n\n"
            f"Analysis: Look for key indicators in the question that point to '{correct_topic}'. "
            f"Consider the specific keywords, user intent, and context that distinguish this category "
            f"from '{predicted_topic}'. Pay attention to domain-specific terminology and the exact "
            f"nature of the customer's request or concern."
        )

    return dspy.Prediction(score=score, feedback=feedback_text)


def save_report(results_dir, results_summary, optimized_instructions, args, n_train, n_val, n_test):
    """Save a human-readable markdown report with all results."""
    
    report_path = os.path.join(results_dir, "REPORT.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate improvement
    baseline = results_summary.get('baseline_accuracy')
    final = results_summary.get('final_accuracy')
    improvement = results_summary.get('improvement')
    
    with open(report_path, 'w') as f:
        f.write("# GEPA Banking Topic Classification - Run Report\n\n")
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


def load_banking_data(max_train=None, max_test=None):
    """Load and process banking dataset into DSPy Examples."""
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
    train_set = [
        dspy.Example({
            "question": x['text'],
            'topic': x['category'],
        }).with_inputs("question")
        for x in train_df.to_dict(orient='records')
    ]
    
    val_set = [
        dspy.Example({
            "question": x['text'],
            'topic': x['category'],
        }).with_inputs("question")
        for x in val_df.to_dict(orient='records')
    ]
    
    test_set = [
        dspy.Example({
            "question": x['text'],
            'topic': x['category'],
        }).with_inputs("question")
        for x in test_df.to_dict(orient='records')
    ]
    
    return train_set, val_set, test_set


def run_baseline_evaluation(test_set, num_threads, results_dir):
    """Step 1: Evaluate baseline (ChainOfThought without optimization) on test set."""
    log("\n" + "="*70)
    log("STEP 1: BASELINE EVALUATION (No Optimization)")
    log("="*70)
    
    # Create baseline program
    baseline_program = dspy.ChainOfThought(TopicResponse)
    
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
    accuracy = evaluate(baseline_program)
    
    # Normalize accuracy to 0-1 range if needed
    if accuracy > 1:
        accuracy = accuracy / 100.0
    
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
    
    return accuracy, baseline_program


def run_gepa_training(train_set, val_set, main_lm, reflection_lm, num_threads, results_dir):
    """Step 2: Train GEPA to optimize the prompt."""
    log("\n" + "="*70)
    log("STEP 2: GEPA TRAINING")
    log("="*70)
    log(f"Training on {len(train_set)} samples, validating on {len(val_set)}")
    
    from dspy import GEPA
    
    # Create a fresh program for optimization
    program_to_optimize = dspy.ChainOfThought(TopicResponse)
    
    # Initialize GEPA optimizer
    gepa_optimizer = GEPA(
        metric=metric_with_feedback,
        auto="light",
        num_threads=num_threads,
        track_stats=True,
        reflection_minibatch_size=16,
        track_best_outputs=True,
        add_format_failure_as_feedback=True,
        reflection_lm=reflection_lm
    )
    
    log(f"\nGEPA configuration:")
    log(f"  - Auto mode: light")
    log(f"  - Num threads: {num_threads}")
    log(f"  - Reflection minibatch size: 16")
    
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
    accuracy = evaluate(optimized_program)
    
    # Normalize accuracy to 0-1 range if needed
    if accuracy > 1:
        accuracy = accuracy / 100.0
    
    log(f"\n📊 Final Test Accuracy (with optimized prompt): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
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
    parser = argparse.ArgumentParser(description='GEPA Banking Workflow')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip initial baseline evaluation')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip GEPA training')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Limit training samples (for faster testing)')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Limit test samples (for faster testing)')
    parser.add_argument('--main-model', type=str, default='anthropic/claude-haiku-4-5-20251001',
                        help='Model to use for main LM')
    parser.add_argument('--reflection-model', type=str, default='anthropic/claude-opus-4-5',
                        help='Model to use for reflection LM')
    parser.add_argument('--num-threads', type=int, default=16,
                        help='Number of threads for parallel evaluation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LM')
    args = parser.parse_args()
    
    log("\n" + "#"*70)
    log("#" + " "*15 + "GEPA BANKING WORKFLOW" + " "*24 + "#")
    log("#"*70)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results", f"gepa_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    log(f"\n📁 Results will be saved to: {results_dir}")
    
    # Initialize LMs
    log(f"\n🤖 Initializing DSPy LMs...")
    log(f"  - Main model: {args.main_model}")
    log(f"  - Reflection model: {args.reflection_model}")
    log(f"  - Temperature: {args.temperature}")
    
    main_lm = dspy.LM(
        args.main_model,
        temperature=args.temperature
    )
    
    reflection_lm = dspy.LM(
        args.reflection_model,
        temperature=args.temperature
    )
    
    dspy.configure(lm=main_lm)
    
    # Load data
    log("\n📦 Loading banking dataset...")
    train_set, val_set, test_set = load_banking_data(
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
            train_set, val_set, main_lm, reflection_lm, args.num_threads, results_dir
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
