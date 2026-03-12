#!/usr/bin/env python3
import os
import re
import json
import openai
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

def initialize_clients(api_provider):
    """Initialize separate clients for generator, reflector, and curator"""
    if api_provider == "sambanova":
        # Use SambaNova API
        base_url = "https://api.sambanova.ai/v1"
        api_key = os.getenv('SAMBANOVA_API_KEY', '')
        if not api_key:
            raise ValueError("SambaNova api key not found in environment variables")
    elif api_provider == "together":
        # Use Together API
        base_url = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHER_API_KEY', '')
        if not api_key:
            raise ValueError("Together api key not found in environment variables")
    elif api_provider == "openai":
        # Use OpenAI API
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OpenAI api key not found in environment variables")
    elif api_provider == "commonstack":
        # Use Commonstack API
        base_url = "https://api.commonstack.ai/v1"
        api_key = os.getenv('COMMONSTACK_API_KEY', '')
        if not api_key:
            raise ValueError("Commonstack api key not found in environment variables")
    else:
        raise ValueError(
            f"Invalid api_provider name: {api_provider}. Must be 'sambanova', 'together', 'openai', or 'commonstack'"
        )
        
    generator_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    reflector_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    curator_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    print(f"Using {api_provider} API for all models")
    return generator_client, reflector_client, curator_client

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

def extract_answer(response):
    """Extract final answer from model response"""
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer  
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try to get final answer from JSON style response with regex matching 
        # Try double quotes first
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try single quotes
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Handle JSON format without quotes (for simple expressions)
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up trailing characters
            answer = re.sub(r'[,}]*$', '', answer)
            return answer
        
        # Fallback for "The final answer is: X" pattern with boxed
        final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
        match = re.search(final_answer_pattern, response)
        if match:
            # Extract boxed content starting from this match
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content
        
        # More general pattern for "final answer is X"
        matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
            answer = answer.replace('$', '').strip()
            if answer:
                return answer
        
        return "No final answer found"
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


def evaluate_single_test_sample(args_tuple, data_processor, retriever=None) -> Tuple[Dict, str]:
    """
    Evaluate a single test sample - task-agnostic implementation.
    
    Args:
        args_tuple: Tuple of (index, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode)
        data_processor: DataProcessor instance with answer_is_correct method
        retriever: Optional Retriever instance. When provided, retrieves a per-sample
                   mini playbook instead of using the full playbook.
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        sub_playbook_tokens = None
        if retriever is not None:
            playbook = retriever.retrieve(question, context)
            sub_playbook_tokens = count_tokens(playbook)

        gen_response, bullet_ids, call_info = generator.generate(
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"test_eval_{i}",
            log_dir=log_dir
        )

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        result = {
            "index": i,
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "success": True
        }
        if sub_playbook_tokens is not None:
            result["sub_playbook_tokens"] = sub_playbook_tokens

        return result, None

    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20, 
                      use_json_mode=False, retriever=None) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of test set - task-agnostic implementation.
    
    Args:
        data_processor: DataProcessor instance with answer_is_correct and evaluate_accuracy methods
        generator: Generator instance
        playbook: Current playbook string (used as fallback when retriever is None)
        test_samples: List of test samples
        max_tokens: Max tokens for generation
        log_dir: Directory for logs
        max_workers: Number of parallel workers
        use_json_mode: Whether to use JSON mode
        retriever: Optional Retriever instance (already indexed). When provided,
                   each sample gets a per-sample mini playbook via retrieval.
        
    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    if retriever is not None:
        print(f"  Using retriever (top_k={retriever.top_k})")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]

    results = {
        "correct": 0, "total": 0, "no_answer": 0,
        "answers": [], "targets": [], "errors": [],
        "sub_playbook_token_counts": []
    }

    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor, retriever=retriever)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(eval_wrapper, args): args 
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            result, error = future.result()
            
            if error:
                print(error)
                continue

            if result and result["success"]:
                results["correct"] += (1 if result["is_correct"] else 0)
                results["total"] += 1
                results["answers"].append(result["final_answer"])
                results["targets"].append(result["target"])

                if "sub_playbook_tokens" in result:
                    results["sub_playbook_token_counts"].append(result["sub_playbook_tokens"])
                
                if not result["is_correct"]:
                    results["errors"].append({
                        "index": result["index"],
                        "prediction": result["final_answer"],
                        "ground_truth": result["target"]
                    })
                
                if result["final_answer"] == "No final answer found":
                    results["no_answer"] += 1

            if i % 50 == 0:
                curr_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {curr_acc:.3f}")
    
    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(results["answers"], results["targets"])
        
        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"]
        }

        token_counts = results["sub_playbook_token_counts"]
        if token_counts:
            avg_tokens = round(sum(token_counts) / len(token_counts), 1)
            final_results["avg_sub_playbook_tokens"] = avg_tokens
            print(f"\n  Avg sub-playbook tokens: {avg_tokens}")
        
        error_logs = {
            "accuracy": accuracy,
            "errors": results["errors"]
        }
        
        print(f"\n  Final Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")
    else:
        final_results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print(f"\n  No valid results!")
        
    return final_results, error_logs
