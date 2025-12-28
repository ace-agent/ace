#!/usr/bin/env python3
"""
End-to-end test script for the full ACE training process with Anthropic.

This script tests the complete ACE workflow:
1. Data loading and processing
2. ACE system initialization
3. Offline training with train/validation samples
4. Testing before and after training
5. Playbook evolution and improvement

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    
    # Run quick tests only (prompts for offline training)
    python3 test_ace_e2e.py
    
    # Run all tests including offline training (no prompts)
    python3 test_ace_e2e.py --all
    
    # Skip the long offline training test
    python3 test_ace_e2e.py --skip-offline
    
    # Run only the offline training test
    python3 test_ace_e2e.py --offline-only

Results are saved to ./test_results/ directory.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def log(message: str = "", end: str = "\n"):
    """Print with immediate flush for real-time output visibility."""
    print(message, end=end, flush=True)


class SimpleDataProcessor:
    """
    Simple data processor for testing ACE with basic Q&A tasks.
    """
    
    def __init__(self, task_name: str = "math_qa"):
        self.task_name = task_name
    
    def process_task_data(self, raw_data):
        """Process raw data into standard format."""
        return raw_data
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from model response."""
        # Try to extract from JSON format first
        try:
            parsed = json.loads(response)
            if "final_answer" in parsed:
                return str(parsed["final_answer"]).strip().lower()
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Try to find answer after common patterns
        response_lower = response.lower()
        
        # Look for "answer:" or "final answer:" patterns
        for pattern in ["final answer:", "answer:", "the answer is", "result:"]:
            if pattern in response_lower:
                idx = response_lower.find(pattern) + len(pattern)
                answer = response[idx:].strip().split('\n')[0].strip()
                # Clean up common punctuation
                answer = answer.strip('.,!?').strip()
                return answer.lower()
        
        # If response is very short, treat it as the answer
        if len(response.strip()) < 50:
            return response.strip().lower()
        
        # Return first line as fallback
        return response.strip().split('\n')[0].strip().lower()
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Evaluate if the predicted answer matches ground truth."""
        pred = str(predicted).strip().lower()
        gt = str(ground_truth).strip().lower()
        
        # Exact match
        if pred == gt:
            return True
        
        # Check if answer is contained
        if gt in pred or pred in gt:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred.replace(',', ''))
            gt_num = float(gt.replace(',', ''))
            if abs(pred_num - gt_num) < 0.01:
                return True
        except (ValueError, TypeError):
            pass
        
        return False
    
    def evaluate_accuracy(self, predictions: list, ground_truths: list) -> float:
        """Compute accuracy over a list of predictions."""
        if not predictions:
            return 0.0
        
        correct = sum(
            self.answer_is_correct(pred, gt) 
            for pred, gt in zip(predictions, ground_truths)
        )
        return correct / len(predictions)


def create_math_qa_dataset():
    """Create a simple math Q&A dataset for testing."""
    
    train_samples = [
        {
            "question": "What is 15 + 27?",
            "context": "",
            "target": "42"
        },
        {
            "question": "What is 100 - 37?",
            "context": "",
            "target": "63"
        },
        {
            "question": "What is 8 * 7?",
            "context": "",
            "target": "56"
        },
        {
            "question": "What is 144 / 12?",
            "context": "",
            "target": "12"
        },
        {
            "question": "What is 25 + 75?",
            "context": "",
            "target": "100"
        },
        {
            "question": "What is 200 - 88?",
            "context": "",
            "target": "112"
        },
        {
            "question": "What is 9 * 9?",
            "context": "",
            "target": "81"
        },
        {
            "question": "What is 120 / 8?",
            "context": "",
            "target": "15"
        },
    ]
    
    val_samples = [
        {
            "question": "What is 33 + 44?",
            "context": "",
            "target": "77"
        },
        {
            "question": "What is 90 - 25?",
            "context": "",
            "target": "65"
        },
        {
            "question": "What is 6 * 11?",
            "context": "",
            "target": "66"
        },
    ]
    
    test_samples = [
        {
            "question": "What is 55 + 45?",
            "context": "",
            "target": "100"
        },
        {
            "question": "What is 150 - 73?",
            "context": "",
            "target": "77"
        },
        {
            "question": "What is 12 * 12?",
            "context": "",
            "target": "144"
        },
        {
            "question": "What is 180 / 9?",
            "context": "",
            "target": "20"
        },
    ]
    
    return train_samples, val_samples, test_samples


def create_geography_qa_dataset():
    """Create a geography Q&A dataset for testing."""
    
    train_samples = [
        {
            "question": "What is the capital of France?",
            "context": "",
            "target": "Paris"
        },
        {
            "question": "What is the capital of Germany?",
            "context": "",
            "target": "Berlin"
        },
        {
            "question": "What is the capital of Italy?",
            "context": "",
            "target": "Rome"
        },
        {
            "question": "What is the capital of Spain?",
            "context": "",
            "target": "Madrid"
        },
        {
            "question": "What is the capital of Japan?",
            "context": "",
            "target": "Tokyo"
        },
        {
            "question": "What is the capital of Australia?",
            "context": "",
            "target": "Canberra"
        },
    ]
    
    val_samples = [
        {
            "question": "What is the capital of Canada?",
            "context": "",
            "target": "Ottawa"
        },
        {
            "question": "What is the capital of Brazil?",
            "context": "",
            "target": "Brasilia"
        },
    ]
    
    test_samples = [
        {
            "question": "What is the capital of Egypt?",
            "context": "",
            "target": "Cairo"
        },
        {
            "question": "What is the capital of South Korea?",
            "context": "",
            "target": "Seoul"
        },
        {
            "question": "What is the capital of Poland?",
            "context": "",
            "target": "Warsaw"
        },
    ]
    
    return train_samples, val_samples, test_samples


def test_ace_offline_training():
    """Test the full ACE offline training workflow."""
    
    log("\n" + "="*70)
    log("ACE END-TO-END TEST: OFFLINE TRAINING MODE")
    log("="*70)
    
    # Create results directory (permanent location)
    results_dir = os.path.join(os.path.dirname(__file__), "test_results", "offline_training")
    os.makedirs(results_dir, exist_ok=True)
    log(f"\nResults will be saved to: {results_dir}")
    
    try:
        from ace import ACE
        
        # Initialize ACE system with Anthropic
        log("\n[1/5] Initializing ACE system with Anthropic...")
        ace_system = ACE(
            api_provider="anthropic",
            generator_model="claude-3-5-haiku-20241022",
            reflector_model="claude-3-5-haiku-20241022",
            curator_model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            initial_playbook=None,  # Start with empty playbook
            use_bulletpoint_analyzer=False
        )
        log("   ✓ ACE system initialized")
        
        # Create dataset
        log("\n[2/5] Creating test dataset...")
        train_samples, val_samples, test_samples = create_geography_qa_dataset()
        log(f"   ✓ Train samples: {len(train_samples)}")
        log(f"   ✓ Validation samples: {len(val_samples)}")
        log(f"   ✓ Test samples: {len(test_samples)}")
        
        # Create data processor
        log("\n[3/5] Creating data processor...")
        processor = SimpleDataProcessor(task_name="geography_qa")
        log("   ✓ Data processor created")
        
        # Configure training
        log("\n[4/5] Configuring training...")
        config = {
            'num_epochs': 1,
            'max_num_rounds': 2,  # Max reflection rounds
            'curator_frequency': 2,  # Run curator every 2 steps
            'eval_steps': 4,  # Evaluate every 4 steps
            'save_steps': 4,
            'playbook_token_budget': 10000,
            'task_name': 'geography_qa_test',
            'json_mode': False,
            'no_ground_truth': False,
            'save_dir': results_dir,
            'test_workers': 1,  # Single worker for testing
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
        playbook_preview = ace_system.best_playbook[:1000] if len(ace_system.best_playbook) > 1000 else ace_system.best_playbook
        log(playbook_preview)
        if len(ace_system.best_playbook) > 1000:
            log(f"\n... (truncated, total length: {len(ace_system.best_playbook)} chars)")
        log("-" * 50)
        
        # Save final playbook
        playbook_path = os.path.join(results_dir, "final_playbook.txt")
        with open(playbook_path, 'w') as f:
            f.write(ace_system.best_playbook)
        log(f"\n💾 Final playbook saved to: {playbook_path}")
        
        log("\n✅ ACE OFFLINE TRAINING TEST PASSED!")
        return True, results_dir
        
    except Exception as e:
        log(f"\n❌ ACE OFFLINE TRAINING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, results_dir


def test_ace_eval_only():
    """Test ACE evaluation-only mode."""
    
    log("\n" + "="*70)
    log("ACE END-TO-END TEST: EVAL ONLY MODE")
    log("="*70)
    
    # Create results directory (permanent location)
    results_dir = os.path.join(os.path.dirname(__file__), "test_results", "eval_only")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        from ace import ACE
        
        # Initialize with a simple initial playbook
        initial_playbook = """## STRATEGIES & INSIGHTS
- For geography questions about capitals, recall common knowledge about world capitals
- Answer directly and concisely with just the city name

## COMMON MISTAKES TO AVOID
- Don't confuse largest city with capital (e.g., Sydney vs Canberra for Australia)
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
        log("   ✓ ACE system initialized with custom playbook")
        
        # Create test data
        log("\n[2/3] Creating test dataset...")
        _, _, test_samples = create_geography_qa_dataset()
        processor = SimpleDataProcessor(task_name="geography_qa")
        log(f"   ✓ Test samples: {len(test_samples)}")
        
        # Run evaluation
        log("\n[3/3] Running evaluation...")
        config = {
            'task_name': 'geography_eval_test',
            'save_dir': results_dir,
            'test_workers': 1,
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
        
        log("\n✅ ACE EVAL ONLY TEST PASSED!")
        return True, results_dir
        
    except Exception as e:
        log(f"\n❌ ACE EVAL ONLY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, results_dir


def test_single_training_step():
    """Test a single training step to verify the core loop works."""
    
    log("\n" + "="*70)
    log("ACE END-TO-END TEST: SINGLE TRAINING STEP")
    log("="*70)
    
    try:
        from ace import ACE
        
        log("\n[1/4] Initializing ACE system...")
        ace_system = ACE(
            api_provider="anthropic",
            generator_model="claude-3-5-haiku-20241022",
            reflector_model="claude-3-5-haiku-20241022",
            curator_model="claude-3-5-haiku-20241022",
            max_tokens=1024
        )
        log("   ✓ ACE system initialized")
        
        # Test single generation
        log("\n[2/4] Testing Generator...")
        question = "What is the capital of the United Kingdom?"
        playbook = ace_system.playbook
        
        response, bullet_ids, call_info = ace_system.generator.generate(
            question=question,
            playbook=playbook,
            context="",
            reflection="(empty)",
            use_json_mode=False,
            call_id="test_gen_step"
        )
        log(f"   ✓ Generator response received ({len(response)} chars)")
        log(f"   Response preview: {response[:200]}...")
        
        # Test reflector with correct parameters
        log("\n[3/4] Testing Reflector...")
        reflection, bullet_tags, call_info = ace_system.reflector.reflect(
            question=question,
            reasoning_trace=response,
            predicted_answer="London",
            ground_truth="London",
            environment_feedback="CORRECT",
            bullets_used="(none used)",
            use_ground_truth=True,
            call_id="test_reflect_step"
        )
        log(f"   ✓ Reflector response received ({len(reflection)} chars)")
        log(f"   Reflection preview: {reflection[:200]}...")
        
        # Test curator with correct parameters
        log("\n[4/4] Testing Curator...")
        playbook_stats = {
            "total_bullets": 0,
            "sections": {},
            "token_count": 100
        }
        updated_playbook, next_id, operations, call_info = ace_system.curator.curate(
            current_playbook=playbook,
            recent_reflection=reflection,
            question_context=question,
            current_step=1,
            total_samples=10,
            token_budget=10000,
            playbook_stats=playbook_stats,
            use_ground_truth=True,
            call_id="test_curate_step"
        )
        log(f"   ✓ Curator response received ({len(updated_playbook)} chars)")
        log(f"   Operations performed: {len(operations)}")
        
        # Check if playbook was updated
        if updated_playbook != playbook:
            log("   ✓ Playbook was updated!")
            log(f"\n   Updated playbook preview:")
            log("-" * 50)
            log(updated_playbook[:500])
            log("-" * 50)
        else:
            log("   ℹ Playbook unchanged (no new insights)")
        
        log("\n✅ SINGLE TRAINING STEP TEST PASSED!")
        return True
        
    except Exception as e:
        log(f"\n❌ SINGLE TRAINING STEP TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all end-to-end tests."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ACE Framework E2E Tests with Anthropic')
    parser.add_argument('--all', '-a', action='store_true', 
                        help='Run all tests including offline training (no prompts)')
    parser.add_argument('--skip-offline', action='store_true',
                        help='Skip the offline training test')
    parser.add_argument('--offline-only', action='store_true',
                        help='Only run the offline training test')
    args = parser.parse_args()
    
    log("\n" + "#"*70)
    log("#" + " "*20 + "ACE FRAMEWORK E2E TESTS" + " "*20 + "#")
    log("#" + " "*15 + "Testing with Anthropic Provider" + " "*15 + "#")
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
        # Test 1: Single training step (quick sanity check)
        result1 = test_single_training_step()
        results.append(("Single Training Step", result1))
        
        # Test 2: Evaluation only mode
        result2, result_dir2 = test_ace_eval_only()
        results.append(("Eval Only Mode", result2))
        result_dirs.append(result_dir2)
    
    # Test 3: Full offline training (this takes longer)
    if args.skip_offline:
        log("\n⏭️  Skipping offline training test (--skip-offline flag)")
        results.append(("Offline Training", "SKIPPED"))
    elif args.all or args.offline_only:
        log("\n" + "="*70)
        log("Starting Offline Training test (this may take several minutes)...")
        log("="*70)
        result3, result_dir3 = test_ace_offline_training()
        results.append(("Offline Training", result3))
        result_dirs.append(result_dir3)
    else:
        log("\n" + "!"*70)
        log("NOTE: The next test (Offline Training) may take several minutes...")
        log("!"*70)
        
        try:
            proceed = input("\nRun full offline training test? [y/N]: ").strip().lower()
        except EOFError:
            proceed = 'n'  # Default to not running if non-interactive
            log("\n(Non-interactive mode detected, skipping offline training)")
            
        if proceed == 'y':
            result3, result_dir3 = test_ace_offline_training()
            results.append(("Offline Training", result3))
            result_dirs.append(result_dir3)
        else:
            log("Skipping offline training test.")
            log("Tip: Use --all to run all tests without prompts")
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
    
    # Show results location
    if result_dirs:
        log(f"\n📁 Test results saved in:")
        for rd in result_dirs:
            log(f"   - {rd}")
        log("\n   Playbooks can be found in the 'ace_run_*' subdirectories.")
    
    if failed > 0:
        log("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        log("\n🎉 All executed tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
