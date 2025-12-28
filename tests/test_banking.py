#!/usr/bin/env python3
"""
End-to-end test script for ACE training on Banking topic classification.

This script tests the complete ACE workflow for the banking dataset:
1. Data loading and processing from CSV files
2. ACE system initialization
3. Offline training with train/validation samples
4. Testing before and after training
5. Playbook evolution for banking topic classification

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    cd /path/to/ace
    
    # Run quick tests only (prompts for offline training)
    python3 -m tests.test_banking
    
    # Run all tests including offline training (no prompts)
    python3 -m tests.test_banking --all
    
    # Skip the long offline training test
    python3 -m tests.test_banking --skip-offline
    
    # Run only the offline training test
    python3 -m tests.test_banking --offline-only
    
    # Use a smaller subset for faster testing
    python3 -m tests.test_banking --offline-only --max-samples 20

Results are saved to ./test_results/banking/ directory.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


def log(message: str = "", end: str = "\n"):
    """Print with immediate flush for real-time output visibility."""
    print(message, end=end, flush=True)


# Import banking data processor
from banking.data_processor import DataProcessor, load_data, ALLOWED_TOPICS


class BankingTestProcessor(DataProcessor):
    """
    Extended processor for testing with additional logging.
    """
    
    def __init__(self, task_name: str = "banking"):
        super().__init__(task_name)
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from model response with banking-specific logic."""
        from banking.data_processor import extract_topic_from_response
        return extract_topic_from_response(response)


def load_banking_dataset(max_samples: int = None):
    """
    Load the banking dataset from CSV files.
    
    Args:
        max_samples: Maximum number of samples to load per split (for quick testing)
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # Go up one level from tests/ to ace/
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "banking", "data")
    
    # Load raw data
    train_raw = load_data(os.path.join(data_path, "train.csv"))
    val_raw = load_data(os.path.join(data_path, "val.csv"))
    test_raw = load_data(os.path.join(data_path, "test.csv"))
    
    # Limit samples if specified
    if max_samples:
        train_raw = train_raw[:max_samples]
        val_raw = val_raw[:min(max_samples // 4, len(val_raw))]
        test_raw = test_raw[:min(max_samples // 2, len(test_raw))]
    
    # Process data
    processor = DataProcessor(task_name="banking")
    train_samples = processor.process_task_data(train_raw)
    val_samples = processor.process_task_data(val_raw)
    test_samples = processor.process_task_data(test_raw)
    
    return train_samples, val_samples, test_samples


def test_data_loading():
    """Test that data loads correctly."""
    log("\n" + "="*70)
    log("TEST: DATA LOADING")
    log("="*70)
    
    try:
        train, val, test = load_banking_dataset(max_samples=10)
        
        log(f"\n✓ Loaded {len(train)} training samples")
        log(f"✓ Loaded {len(val)} validation samples")
        log(f"✓ Loaded {len(test)} test samples")
        
        # Show sample
        log("\n📋 Sample processed data:")
        sample = train[0]
        log(f"   Question (truncated): {sample['question'][:200]}...")
        log(f"   Target: {sample['target']}")
        
        log("\n✅ DATA LOADING TEST PASSED!")
        return True
        
    except Exception as e:
        log(f"\n❌ DATA LOADING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_answer_extraction():
    """Test the answer extraction logic."""
    log("\n" + "="*70)
    log("TEST: ANSWER EXTRACTION")
    log("="*70)
    
    try:
        from banking.data_processor import extract_topic_from_response, normalize_topic
        
        test_cases = [
            # Simple cases
            ("declined_card_payment", "declined_card_payment"),
            ("DECLINED_CARD_PAYMENT", "declined_card_payment"),
            ("declined card payment", "declined_card_payment"),
            
            # With explanation
            ("The topic is: card_arrival", "card_arrival"),
            ("Answer: top_up_failed", "top_up_failed"),
            ("Based on the query, the category is exchange_rate", "exchange_rate"),
            
            # JSON format
            ('{"topic": "pin_blocked"}', "pin_blocked"),
            ('{"answer": "lost_or_stolen_card"}', "lost_or_stolen_card"),
        ]
        
        passed = 0
        for response, expected in test_cases:
            result = extract_topic_from_response(response)
            expected_norm = normalize_topic(expected)
            
            if result == expected_norm:
                log(f"   ✓ '{response[:40]}...' -> '{result}'")
                passed += 1
            else:
                log(f"   ✗ '{response[:40]}...' -> '{result}' (expected: '{expected_norm}')")
        
        log(f"\n   Passed {passed}/{len(test_cases)} extraction tests")
        
        if passed >= len(test_cases) * 0.8:  # 80% pass rate
            log("\n✅ ANSWER EXTRACTION TEST PASSED!")
            return True
        else:
            log("\n❌ ANSWER EXTRACTION TEST FAILED!")
            return False
        
    except Exception as e:
        log(f"\n❌ ANSWER EXTRACTION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_accuracy_evaluation():
    """Test the accuracy evaluation logic."""
    log("\n" + "="*70)
    log("TEST: ACCURACY EVALUATION")
    log("="*70)
    
    try:
        processor = DataProcessor(task_name="banking")
        
        # Test cases
        predictions = [
            "declined_card_payment",
            "CARD_ARRIVAL",  # Should match card_arrival
            "top up failed",  # Should match top_up_failed
            "wrong_topic",  # Should not match
            "exchange_rate",
        ]
        ground_truths = [
            "declined_card_payment",
            "card_arrival",
            "top_up_failed",
            "pin_blocked",
            "exchange_rate",
        ]
        
        accuracy = processor.evaluate_accuracy(predictions, ground_truths)
        expected_accuracy = 4 / 5  # 4 correct out of 5
        
        log(f"\n   Predictions: {predictions}")
        log(f"   Ground truths: {ground_truths}")
        log(f"   Calculated accuracy: {accuracy:.2f}")
        log(f"   Expected accuracy: {expected_accuracy:.2f}")
        
        if abs(accuracy - expected_accuracy) < 0.01:
            log("\n✅ ACCURACY EVALUATION TEST PASSED!")
            return True
        else:
            log("\n❌ ACCURACY EVALUATION TEST FAILED!")
            return False
        
    except Exception as e:
        log(f"\n❌ ACCURACY EVALUATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_generation():
    """Test a single generation step with ACE."""
    log("\n" + "="*70)
    log("TEST: SINGLE GENERATION STEP")
    log("="*70)
    
    try:
        from ace import ACE
        
        log("\n[1/3] Initializing ACE system...")
        ace_system = ACE(
            api_provider="anthropic",
            generator_model="claude-3-5-haiku-20241022",
            reflector_model="claude-3-5-haiku-20241022",
            curator_model="claude-3-5-haiku-20241022",
            max_tokens=1024
        )
        log("   ✓ ACE system initialized")
        
        # Load a sample
        log("\n[2/3] Loading sample data...")
        train, _, _ = load_banking_dataset(max_samples=5)
        sample = train[0]
        log(f"   ✓ Sample loaded")
        log(f"   Target topic: {sample['target']}")
        
        # Generate response
        log("\n[3/3] Testing generator...")
        response, bullet_ids, call_info = ace_system.generator.generate(
            question=sample['question'],
            playbook=ace_system.playbook,
            context=sample['context'],
            reflection="(empty)",
            use_json_mode=False,
            call_id="test_banking_gen"
        )
        
        log(f"   ✓ Response received ({len(response)} chars)")
        log(f"   Response preview: {response[:300]}...")
        
        # Extract and check answer
        from banking.data_processor import extract_topic_from_response
        extracted = extract_topic_from_response(response)
        log(f"   Extracted topic: {extracted}")
        
        processor = DataProcessor(task_name="banking")
        is_correct = processor.answer_is_correct(extracted, sample['target'])
        log(f"   Is correct: {is_correct}")
        
        log("\n✅ SINGLE GENERATION TEST PASSED!")
        return True
        
    except Exception as e:
        log(f"\n❌ SINGLE GENERATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ace_offline_training(max_samples: int = None):
    """Test the full ACE offline training workflow."""
    
    log("\n" + "="*70)
    log("ACE BANKING TEST: OFFLINE TRAINING MODE")
    log("="*70)
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "test_results", "banking_offline")
    os.makedirs(results_dir, exist_ok=True)
    log(f"\nResults will be saved to: {results_dir}")
    
    try:
        from ace import ACE
        
        # Initialize ACE system
        log("\n[1/5] Initializing ACE system with Anthropic...")
        ace_system = ACE(
            api_provider="anthropic",
            generator_model="claude-3-5-haiku-20241022",
            reflector_model="claude-3-5-haiku-20241022",
            curator_model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            initial_playbook=None,
            use_bulletpoint_analyzer=False
        )
        log("   ✓ ACE system initialized")
        
        # Load dataset
        log("\n[2/5] Loading banking dataset...")
        train_samples, val_samples, test_samples = load_banking_dataset(max_samples=max_samples)
        log(f"   ✓ Train samples: {len(train_samples)}")
        log(f"   ✓ Validation samples: {len(val_samples)}")
        log(f"   ✓ Test samples: {len(test_samples)}")
        
        # Create data processor
        log("\n[3/5] Creating data processor...")
        processor = BankingTestProcessor(task_name="banking")
        log("   ✓ Data processor created")
        
        # Configure training
        log("\n[4/5] Configuring training...")
        config = {
            'num_epochs': 1,
            'max_num_rounds': 2,
            'curator_frequency': 2,
            'eval_steps': max(len(train_samples) // 2, 5),
            'save_steps': max(len(train_samples) // 2, 5),
            'playbook_token_budget': 20000,
            'task_name': 'banking_test',
            'json_mode': False,
            'no_ground_truth': False,
            'save_dir': results_dir,
            'test_workers': 2,
        }
        log(f"   ✓ Configuration:")
        for key, value in config.items():
            log(f"      - {key}: {value}")
        
        # Run offline training
        log("\n[5/5] Starting offline training...")
        log("      (ACE system logs will appear below)")
        log("-" * 70)
        
        results = ace_system.run(
            mode='offline',
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            data_processor=processor,
            config=config
        )
        
        log("-" * 70)
        log("\n" + "="*70)
        log("TRAINING COMPLETED")
        log("="*70)
        
        # Display results
        log("\n📊 RESULTS SUMMARY:")
        
        if 'initial_test_results' in results:
            log(f"\n   Initial Test Accuracy: {results['initial_test_results'].get('accuracy', 'N/A'):.3f}")
        
        if 'training_results' in results:
            tr = results['training_results']
            log(f"\n   Training Results:")
            log(f"      - Final validation accuracy: {tr.get('final_val_accuracy', 'N/A')}")
            log(f"      - Best validation accuracy: {tr.get('best_val_accuracy', 'N/A')}")
            log(f"      - Training steps completed: {tr.get('steps_completed', 'N/A')}")
        
        if 'final_test_results' in results:
            log(f"\n   Final Test Accuracy: {results['final_test_results'].get('accuracy', 'N/A'):.3f}")
        
        # Show playbook evolution
        log("\n📝 FINAL PLAYBOOK:")
        log("-" * 50)
        playbook_preview = ace_system.best_playbook[:1500] if len(ace_system.best_playbook) > 1500 else ace_system.best_playbook
        log(playbook_preview)
        if len(ace_system.best_playbook) > 1500:
            log(f"\n... (truncated, total length: {len(ace_system.best_playbook)} chars)")
        log("-" * 50)
        
        # Save final playbook
        playbook_path = os.path.join(results_dir, "final_playbook.txt")
        with open(playbook_path, 'w') as f:
            f.write(ace_system.best_playbook)
        log(f"\n💾 Final playbook saved to: {playbook_path}")
        
        # Save results
        results_path = os.path.join(results_dir, "results.json")
        with open(results_path, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                         for k, v in value.items()}
                else:
                    json_results[key] = str(value)
            json.dump(json_results, f, indent=2)
        log(f"💾 Results saved to: {results_path}")
        
        log("\n✅ ACE BANKING OFFLINE TRAINING TEST PASSED!")
        return True, results_dir
        
    except Exception as e:
        log(f"\n❌ ACE BANKING OFFLINE TRAINING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, results_dir


def test_ace_eval_only():
    """Test ACE evaluation-only mode with a banking-specific playbook."""
    
    log("\n" + "="*70)
    log("ACE BANKING TEST: EVAL ONLY MODE")
    log("="*70)
    
    results_dir = os.path.join(os.path.dirname(__file__), "test_results", "banking_eval")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        from ace import ACE
        
        # Banking-specific initial playbook
        initial_playbook = """## STRATEGIES & INSIGHTS FOR BANKING TOPIC CLASSIFICATION

### Key Topic Categories
- **Card-related issues**: card_arrival, card_delivery_estimate, card_not_working, declined_card_payment, lost_or_stolen_card, compromised_card, activate_my_card, card_about_to_expire
- **Payment issues**: pending_card_payment, card_payment_not_recognised, card_payment_fee_charged, transaction_charged_twice, reverted_card_payment?
- **Transfer issues**: pending_transfer, failed_transfer, declined_transfer, transfer_not_received_by_recipient, cancel_transfer, transfer_timing, transfer_fee_charged
- **Top-up issues**: top_up_failed, pending_top_up, top_up_reverted, top_up_limits, topping_up_by_card
- **Account/Identity**: verify_my_identity, why_verify_identity, unable_to_verify_identity, terminate_account, edit_personal_details

### Classification Tips
- Look for keywords: "card", "payment", "transfer", "top up", "exchange", "ATM", "PIN"
- "Declined" usually indicates a declined_* category
- "Pending" usually indicates a pending_* category
- Questions about delivery/arrival relate to card_delivery_estimate or card_arrival

## COMMON MISTAKES TO AVOID
- Don't confuse card_arrival (asking about card status) with card_delivery_estimate (asking how long)
- Don't confuse pending_card_payment with declined_card_payment
"""
        
        log("\n[1/3] Initializing ACE system...")
        ace_system = ACE(
            api_provider="anthropic",
            generator_model="claude-3-5-haiku-20241022",
            reflector_model="claude-3-5-haiku-20241022",
            curator_model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            initial_playbook=initial_playbook
        )
        log("   ✓ ACE system initialized with banking playbook")
        
        # Load test data (small subset)
        log("\n[2/3] Loading test dataset...")
        _, _, test_samples = load_banking_dataset(max_samples=20)
        processor = BankingTestProcessor(task_name="banking")
        log(f"   ✓ Test samples: {len(test_samples)}")
        
        # Run evaluation
        log("\n[3/3] Running evaluation...")
        config = {
            'task_name': 'banking_eval_test',
            'save_dir': results_dir,
            'test_workers': 2,
            'json_mode': False,
        }
        
        results = ace_system.run(
            mode='eval_only',
            test_samples=test_samples,
            data_processor=processor,
            config=config
        )
        
        log("\n📊 EVALUATION RESULTS:")
        if 'test_results' in results:
            log(f"   Test Accuracy: {results['test_results'].get('accuracy', 'N/A'):.3f}")
        
        log("\n✅ ACE BANKING EVAL ONLY TEST PASSED!")
        return True, results_dir
        
    except Exception as e:
        log(f"\n❌ ACE BANKING EVAL ONLY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, results_dir


def main():
    """Run all banking-specific end-to-end tests."""
    
    parser = argparse.ArgumentParser(description='ACE Framework Banking E2E Tests')
    parser.add_argument('--all', '-a', action='store_true', 
                        help='Run all tests including offline training (no prompts)')
    parser.add_argument('--skip-offline', action='store_true',
                        help='Skip the offline training test')
    parser.add_argument('--offline-only', action='store_true',
                        help='Only run the offline training test')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per split for faster testing (e.g., 20)')
    args = parser.parse_args()
    
    log("\n" + "#"*70)
    log("#" + " "*15 + "ACE FRAMEWORK BANKING E2E TESTS" + " "*15 + "#")
    log("#" + " "*12 + "Testing with Anthropic Provider" + " "*17 + "#")
    log("#"*70)
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if not api_key:
        log("\n❌ ERROR: ANTHROPIC_API_KEY environment variable not set!")
        log("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)
    
    log(f"\n✓ ANTHROPIC_API_KEY is set (length: {len(api_key)} chars)")
    
    results = []
    result_dirs = []
    
    if not args.offline_only:
        # Test 1: Data loading
        result1 = test_data_loading()
        results.append(("Data Loading", result1))
        
        # Test 2: Answer extraction
        result2 = test_answer_extraction()
        results.append(("Answer Extraction", result2))
        
        # Test 3: Accuracy evaluation
        result3 = test_accuracy_evaluation()
        results.append(("Accuracy Evaluation", result3))
        
        # Test 4: Single generation (requires API)
        result4 = test_single_generation()
        results.append(("Single Generation", result4))
        
        # Test 5: Eval only mode
        result5, result_dir5 = test_ace_eval_only()
        results.append(("Eval Only Mode", result5))
        result_dirs.append(result_dir5)
    
    # Test 6: Full offline training
    if args.skip_offline:
        log("\n⏭️  Skipping offline training test (--skip-offline flag)")
        results.append(("Offline Training", "SKIPPED"))
    elif args.all or args.offline_only:
        log("\n" + "="*70)
        log("Starting Offline Training test...")
        if args.max_samples:
            log(f"Using max {args.max_samples} samples per split for faster testing")
        log("="*70)
        result6, result_dir6 = test_ace_offline_training(max_samples=args.max_samples)
        results.append(("Offline Training", result6))
        result_dirs.append(result_dir6)
    else:
        log("\n" + "!"*70)
        log("NOTE: The offline training test may take several minutes...")
        log("!"*70)
        
        try:
            proceed = input("\nRun full offline training test? [y/N]: ").strip().lower()
        except EOFError:
            proceed = 'n'
            log("\n(Non-interactive mode detected, skipping offline training)")
            
        if proceed == 'y':
            result6, result_dir6 = test_ace_offline_training(max_samples=args.max_samples)
            results.append(("Offline Training", result6))
            result_dirs.append(result_dir6)
        else:
            log("Skipping offline training test.")
            log("Tip: Use --all to run all tests without prompts")
            log("Tip: Use --max-samples 20 for faster testing")
            results.append(("Offline Training", "SKIPPED"))
    
    # Summary
    log("\n" + "="*70)
    log("TEST SUMMARY")
    log("="*70)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result == "SKIPPED":
            status = "⏭️  SKIP"
            skipped += 1
        elif result:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        log(f"  {status}: {name}")
    
    log(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if result_dirs:
        log(f"\n📁 Test results saved in:")
        for rd in result_dirs:
            log(f"   - {rd}")
    
    if failed > 0:
        log("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        log("\n🎉 All executed tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
