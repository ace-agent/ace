"""
Data processor for Python code generation task using MBPP dataset.

This processor handles the coding dataset where:
- Input: Natural language problem description + test cases
- Output: Python code solution that passes all test cases
"""

import os
import sys
import csv
import re
import ast
import signal
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and process data from a CSV file.
    
    Expected CSV format: text,code,test_list,group (with header)
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'text': row['text'],
                'code': row['code'],
                'test_list': row['test_list'],
                'group': row.get('group', '')
            })
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from a model response.
    
    The model might return the code in various formats:
    - Just the code
    - Wrapped in ```python ... ``` markdown
    - With explanations before/after
    - In JSON format
    - With escaped newlines (\\n instead of actual newlines)
    
    Args:
        response: Raw model response
        
    Returns:
        Extracted Python code
    """
    import json
    
    def safe_unescape(text: str) -> str:
        """Safely decode unicode escape sequences, handling edge cases."""
        try:
            # Try direct decode
            return text.encode().decode('unicode_escape')
        except UnicodeDecodeError:
            # If decode fails (e.g., backslash at end), try manual replacement
            # Replace common escape sequences manually
            text = text.replace('\\n', '\n')
            text = text.replace('\\t', '\t')
            text = text.replace('\\r', '\r')
            text = text.replace("\\'", "'")
            text = text.replace('\\"', '"')
            text = text.replace('\\\\', '\\')
            return text
    
    response = response.strip()
    
    # CRITICAL FIX: Handle escaped newlines (\\n as literal text)
    # This happens when the LLM returns code with escaped characters
    # Check if response contains literal \\n patterns (escaped newlines)
    if '\\n' in response and '\n' not in response:
        # Response has only escaped newlines, decode them
        response = safe_unescape(response)
    elif '\\n' in response:
        # Mixed case - try to intelligently handle it
        # Count actual vs escaped newlines
        actual_newlines = response.count('\n')
        escaped_newlines = response.count('\\n')
        # If there are more escaped newlines, it's likely they should be decoded
        if escaped_newlines > actual_newlines * 2:
            response = safe_unescape(response)
    
    # Try JSON parsing first
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            for key in ['code', 'solution', 'answer', 'python_code', 'function']:
                if key in parsed:
                    code = str(parsed[key]).strip()
                    # Also decode escaped newlines in JSON responses
                    if '\\n' in code:
                        code = safe_unescape(code)
                    return code
    except (json.JSONDecodeError, KeyError):
        pass
    
    # Try to extract code from markdown code blocks
    # Match ```python...``` or ```...```
    code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Return the first code block found
        code = matches[0].strip()
        # Decode escaped newlines if present
        if '\\n' in code:
            code = safe_unescape(code)
        return code

    # Remove markdown code block markers only from the beginning and end
    code = response.strip()
    # Remove leading ```python (with optional whitespace)
    code = re.sub(r'^```python\s*\n?', '', code)
    # Remove trailing ``` (with optional whitespace/newlines)
    code = re.sub(r'```[\s\n]*$', '', code)
    code = code.strip()
    
    # Check for common prefixes and extract code after them
    for pattern in ['Here is the code:', 'Here\'s the solution:', 'Solution:', 'Code:']:
        if pattern in response:
            idx = response.find(pattern) + len(pattern)
            code_part = response[idx:].strip()
            # If there's still markdown, try to extract it
            matches = re.findall(code_block_pattern, code_part, re.DOTALL)
            if matches:
                extracted = matches[0].strip()
                if '\\n' in extracted:
                    extracted = safe_unescape(extracted)
                return extracted
            # Decode escaped newlines if present
            if '\\n' in code_part:
                code_part = safe_unescape(code_part)
            return code_part
    
    # If response looks like pure code (starts with def, import, class, etc.)
    # or is short enough to be just code, return as-is
    code_start_patterns = ['def ', 'import ', 'from ', 'class ', '@']
    first_line = response.strip().split('\n')[0] if '\n' in response else response.strip()
    if any(first_line.startswith(p) for p in code_start_patterns):
        return response.strip()
    
    # Last resort: if response contains 'def ' anywhere, extract from there to end
    if 'def ' in response:
        idx = response.find('def ')
        return response[idx:].strip()
    
    # Return the response as-is
    return code.strip()


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """
    Context manager to limit execution time.
    Thread-safe version that works in multi-threaded environments like GEPA.
    
    Uses multiprocessing for true timeout enforcement that works in all threads.
    """
    import threading
    import multiprocessing
    import queue
    
    # Check if we're in the main thread
    is_main_thread = threading.current_thread() == threading.main_thread()
    
    if is_main_thread:
        # Use signal-based timeout in main thread (most efficient)
        import signal
        
        def signal_handler(signum, frame):
            raise TimeoutException("Code execution timed out")
        
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # In worker threads, we can't use signals. 
        # We'll use a simple approach: just yield and rely on the outer
        # process-based execution with timeout in execute_code_with_tests.
        # The actual timeout is enforced there using multiprocessing.
        yield


def parse_test_list(test_list_str: str) -> List[str]:
    """
    Parse the test_list string into individual test assertions.
    
    Args:
        test_list_str: String representation of test list (e.g., "['assert func(1)==2', ...]")
        
    Returns:
        List of test assertion strings
    """
    try:
        # Try to parse as Python literal
        tests = ast.literal_eval(test_list_str)
        if isinstance(tests, list):
            return tests
    except (ValueError, SyntaxError):
        pass
    
    # Fallback: try to extract assert statements manually
    # This handles cases where the string might be malformed
    tests = []
    # Split by 'assert' and clean up
    parts = test_list_str.split('assert ')
    for part in parts[1:]:  # Skip first empty part
        # Clean up quotes and brackets
        test = 'assert ' + part.strip().strip("',\"[]")
        if test.strip():
            tests.append(test)
    
    return tests


def _run_code_directly(code: str, test_list: List[str]) -> Dict[str, Any]:
    """
    Helper function to run code directly in the current process.
    """
    result = {
        'success': False,
        'passed': 0,
        'total': len(test_list),
        'errors': [],
        'output': '',
        'timeout': False
    }
    
    # Create isolated execution environment
    exec_globals = {
        '__builtins__': __builtins__,
    }
    
    try:
        # Execute the code to define functions/classes
        exec(code, exec_globals)
        
        # Run each test assertion
        for i, test in enumerate(test_list):
            try:
                exec(test, exec_globals)
                result['passed'] += 1
            except AssertionError as e:
                result['errors'].append(f"Test {i+1} failed: {test}")
            except Exception as e:
                result['errors'].append(f"Test {i+1} error: {test} - {type(e).__name__}: {str(e)}")
        
        # Check if all tests passed
        result['success'] = (result['passed'] == result['total'])
        
    except SyntaxError as e:
        result['errors'].append(f"Syntax error: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Execution error: {type(e).__name__}: {str(e)}")
    
    return result


def execute_code_with_tests(code: str, test_list: List[str], timeout: int = 5) -> Dict[str, Any]:
    """
    Execute generated code and run test assertions.
    
    Uses subprocess for reliable timeout that works in all threads.
    
    Args:
        code: Python code to execute
        test_list: List of test assertion strings
        timeout: Maximum execution time in seconds
        
    Returns:
        {
            'success': bool,           # All tests passed
            'passed': int,             # Number of tests passed
            'total': int,              # Total number of tests
            'errors': list[str],       # Error messages
            'output': str,             # Execution output
            'timeout': bool            # Whether execution timed out
        }
    """
    import subprocess
    import json
    import tempfile
    import threading
    
    result = {
        'success': False,
        'passed': 0,
        'total': len(test_list),
        'errors': [],
        'output': '',
        'timeout': False
    }
    
    if not code.strip():
        result['errors'].append("Empty code provided")
        return result
    
    if not test_list:
        result['errors'].append("No tests provided")
        return result
    
    # Check if we're in the main thread - if so, we can use signal-based timeout
    is_main_thread = threading.current_thread() == threading.main_thread()
    
    if is_main_thread:
        # Use signal-based timeout in main thread (faster, no subprocess overhead)
        try:
            with time_limit(timeout):
                result = _run_code_directly(code, test_list)
        except TimeoutException:
            result['timeout'] = True
            result['errors'].append(f"Code execution timed out (>{timeout}s)")
    else:
        # In worker threads, use subprocess with timeout for reliable termination
        # This is slower but guarantees the process can be killed on timeout
        # Create a Python script that executes the code and tests
        
        # Escape the code properly for embedding in a Python script
        import base64
        code_b64 = base64.b64encode(code.encode()).decode()
        test_list_json = json.dumps(test_list)
        
        test_script = f'''
import sys
import json
import base64

code = base64.b64decode("{code_b64}").decode()
test_list = {test_list_json}

result = {{
    'success': False,
    'passed': 0,
    'total': len(test_list),
    'errors': [],
    'output': '',
    'timeout': False
}}

exec_globals = {{'__builtins__': __builtins__}}

try:
    exec(code, exec_globals)
    
    for i, test in enumerate(test_list):
        try:
            exec(test, exec_globals)
            result['passed'] += 1
        except AssertionError:
            result['errors'].append(f"Test {{i+1}} failed: {{test}}")
        except Exception as e:
            result['errors'].append(f"Test {{i+1}} error: {{test}} - {{type(e).__name__}}: {{str(e)}}")
    
    result['success'] = (result['passed'] == result['total'])
    
except SyntaxError as e:
    result['errors'].append(f"Syntax error: {{str(e)}}")
except Exception as e:
    result['errors'].append(f"Execution error: {{type(e).__name__}}: {{str(e)}}")

print(json.dumps(result))
'''
        
        try:
            proc = subprocess.run(
                [sys.executable, '-c', test_script],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if proc.returncode == 0 and proc.stdout.strip():
                try:
                    result = json.loads(proc.stdout.strip())
                except json.JSONDecodeError:
                    result['errors'].append(f"Failed to parse output: {proc.stdout[:200]}")
            else:
                if proc.stderr:
                    result['errors'].append(f"Execution error: {proc.stderr[:500]}")
                    
        except subprocess.TimeoutExpired:
            result['timeout'] = True
            result['errors'].append(f"Code execution timed out (>{timeout}s)")
        except Exception as e:
            # Fallback to direct execution if subprocess fails
            result = _run_code_directly(code, test_list)
    
    # Generate output summary
    if result['success']:
        result['output'] = f"✓ All {result['total']} tests passed"
    else:
        result['output'] = f"✗ {result['passed']}/{result['total']} tests passed"
        if result['errors']:
            result['output'] += f"\nErrors:\n" + "\n".join(result['errors'])
    
    return result


class DataProcessor:
    """
    Processor for handling Python code generation task.
    
    This processor:
    1. Converts raw CSV data to standardized format
    2. Executes generated code against test cases
    3. Calculates code generation accuracy
    """
    
    def __init__(self, task_name: str = "coding"):
        """
        Initialize the data processor.
        
        Args:
            task_name: The name of the task (default: "coding")
        """
        self.task_name = task_name
        self.timeout = 5  # seconds per test execution
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw CSV data into standardized format for ACE.
        
        Args:
            raw_data: Raw data loaded from CSV (list of dicts with 'text', 'code', 'test_list')
            
        Returns:
            List of dicts with keys: 'context', 'question', 'target'
        """
        processed_data = []
        
        for item in raw_data:
            problem_text = item.get('text', '')
            ground_truth_code = item.get('code', '')
            test_list_str = item.get('test_list', '')
            
            # Parse test list
            test_list = parse_test_list(test_list_str)
            test_cases_formatted = '\n'.join(test_list)
            
            # The question provides the code generation task instruction
            question = (
                f"Write a Python function to solve the following problem:\n\n"
                f"Problem: {problem_text}\n\n"
                f"Your code must pass the following test cases:\n"
                f"{test_cases_formatted}\n\n"
                f"Important: The test cases will be executed against your code. "
                f"Make sure your function name and signature match what the tests expect.\n\n"
                f"Respond with ONLY the Python code, no explanations."
            )
            
            processed_item = {
                "context": "",  # No additional context needed
                "question": question,
                "target": ground_truth_code,
                "others": {
                    "original_text": problem_text,
                    "test_list": test_list,
                    "task": self.task_name,
                }
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the generated code from model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Extracted Python code string
        """
        return extract_code_from_response(response)
    
    def answer_is_correct(self, predicted: str, ground_truth: str, test_list: List[str] = None, 
                          idx: int = None, save_dir: str = None) -> bool:
        """
        Check if the predicted code is correct by executing it with test cases.
        
        Args:
            predicted: Model's predicted code
            ground_truth: Ground truth code (not used for evaluation, only reference)
            test_list: List of test assertions to run
            idx: Index of the sample (for logging)
            save_dir: Directory to save execution logs (defaults to ./logs/execution)
            
        Returns:
            bool: True if all tests pass, False otherwise
        """
        if test_list is None:
            # If no test list provided, we can't evaluate
            return False
        
        # Extract code from response if needed
        code = extract_code_from_response(predicted)
        # Execute code with tests
        result = execute_code_with_tests(code, test_list, timeout=self.timeout)
        
        # Log execution results to a better location
        # Always log, not just failures - this helps with debugging
        if save_dir is None:
            save_dir = 'logs/execution'
        
        import os
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create separate subdirectories for passes and failures
        status = 'success' if result['success'] else 'failure'
        status_dir = os.path.join(save_dir, status)
        os.makedirs(status_dir, exist_ok=True)
        
        # Generate timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sample_id = idx if idx is not None else "unknown"
        log_file = os.path.join(status_dir, f'sample_{sample_id}_{timestamp}.txt')
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('='*70 + '\n')
            f.write(f'Execution Log - Sample {sample_id}\n')
            f.write(f'Status: {status.upper()}\n')
            f.write(f'Timestamp: {timestamp}\n')
            f.write('='*70 + '\n\n')
            
            f.write('--- RAW LLM RESPONSE ---\n')
            if isinstance(predicted, str):
                f.write(predicted)
                if not predicted.endswith('\n'):
                    f.write('\n')
            else:
                f.write(str(predicted) + '\n')
            
            f.write('\n--- EXTRACTED CODE ---\n')
            f.write(code)
            if not code.endswith('\n'):
                f.write('\n')
            
            f.write('\n--- TEST CASES ---\n')
            for i, test in enumerate(test_list, 1):
                f.write(f'{i}. {test}\n')
            
            f.write('\n--- EXECUTION RESULT ---\n')
            f.write(f"Success: {result['success']}\n")
            f.write(f"Tests Passed: {result.get('passed', 0)}/{result.get('total', 0)}\n")
            f.write(f"Timeout: {result.get('timeout', False)}\n")
            
            if result.get('errors'):
                f.write('\nErrors:\n')
                for error in result['errors']:
                    f.write(f'  - {error}\n')
            
            if result.get('output'):
                f.write(f'\nOutput:\n{result["output"]}\n')
            
            f.write('\n--- GROUND TRUTH (Reference) ---\n')
            f.write(ground_truth)
            if not ground_truth.endswith('\n'):
                f.write('\n')
        
        return result['success']
    
    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str], 
                         test_lists: List[List[str]], save_dir: str = None) -> float:
        """
        Calculate code generation accuracy across multiple predictions.
        
        Args:
            predictions: List of model predictions (code)
            ground_truths: List of ground truth code (reference only)
            test_lists: List of test assertion lists for each problem
            save_dir: Directory to save execution logs (optional)
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) != len(ground_truths) or len(predictions) != len(test_lists):
            raise ValueError("Predictions, ground truths, and test lists must have same length")
        
        if not predictions:
            return 0.0
        
        correct = 0
        for idx, (pred, truth, tests) in enumerate(zip(predictions, ground_truths, test_lists)):
            if self.answer_is_correct(pred, truth, tests, idx=idx, save_dir=save_dir):
                correct += 1
        return correct / len(predictions)
    
    def get_detailed_results(self, predictions: List[str], ground_truths: List[str],
                            test_lists: List[List[str]]) -> Dict:
        """
        Get detailed evaluation results including per-problem execution details.
        
        Args:
            predictions: List of model predictions (code)
            ground_truths: List of ground truth code (reference only)
            test_lists: List of test assertion lists for each problem
            
        Returns:
            Dict with detailed metrics
        """
        results = {
            'total': len(predictions),
            'correct': 0,
            'accuracy': 0.0,
            'per_problem': [],
            'error_types': {
                'syntax_error': 0,
                'runtime_error': 0,
                'test_failure': 0,
                'timeout': 0,
            }
        }
        
        for i, (pred, truth, tests) in enumerate(zip(predictions, ground_truths, test_lists)):
            code = extract_code_from_response(pred)
            execution_result = execute_code_with_tests(code, tests, timeout=self.timeout)
            
            is_correct = execution_result['success']
            
            if is_correct:
                results['correct'] += 1
            
            problem_result = {
                'index': i,
                'correct': is_correct,
                'passed_tests': execution_result['passed'],
                'total_tests': execution_result['total'],
                'errors': execution_result['errors'],
                'timeout': execution_result['timeout']
            }
            results['per_problem'].append(problem_result)
            
            # Categorize errors
            if not is_correct:
                if execution_result['timeout']:
                    results['error_types']['timeout'] += 1
                elif any('Syntax error' in err for err in execution_result['errors']):
                    results['error_types']['syntax_error'] += 1
                elif any('Test' in err and 'failed' in err for err in execution_result['errors']):
                    results['error_types']['test_failure'] += 1
                else:
                    results['error_types']['runtime_error'] += 1
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        
        return results
