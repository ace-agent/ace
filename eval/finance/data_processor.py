import os
import json
from utils import extract_answer
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_data(data_path: str) -> List[Dict[str, Any]]:
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

def parse_instruction_and_input(all_context):
    """Parse context to extract question and context parts for finlora_sentiment dataset
    
    Expected format:
    "Instruction: [INSTRUCTION].\nInput: [TEXT]\nAnswer: "
    """
    if "Input: " in all_context and "Instruction: " in all_context:
        # Split by "Input: " to separate instruction from input text
        instruction_part = all_context.split("Input: ")[0].strip()
        instruction_part = instruction_part.split("Instruction: ")[1].strip()
                
        remaining = all_context.split("Input: ")[1]
        input_text = remaining.split("Answer: ")[0].strip()
        return input_text, instruction_part
    
    return "", all_context


def parse_context_and_question_formula(all_context):
    """Parse context to extract question and context parts for formula dataset
    
    Expected format:
    "[some instruction] Question: \"[QUESTION TEXT]\". Answer:"
    """
    if "Question: " in all_context and ". Answer:" in all_context:
        # Split by "Question: " to separate instruction from question
        parts = all_context.split("Question: ", 1)
        instruction_part = parts[0].strip()
        
        # Extract question text (between "Question: " and ". Answer:")
        question_part = parts[1]
        question_text = question_part.split(". Answer:")[0].strip()
        # Remove quotes if present
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += " Your answer should be a plain floating point number, round to the nearest hundredth if necessary. Do the necessary conversions, for example 5 million should be 5000000.0. "
        return "", question_text

    return "", all_context

class DataProcessor:
    """
    Processor for handling data preprocessing, evaluation and accuracy computation.
    """
    
    def __init__(self, task_name: str):
        """
        Initialize the data processor.
        
        Args:
            task_name: The name of the task
        """
        self.task_name = task_name
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Process task dataset format.

        Args:
            raw_data: Raw data from JSONL file
            parse_fn: data parsing function 

        Returns:
            Processed data in standard format
        """
        processed_data = []
        if self.task_name == "finer":
            parse_fn = parse_instruction_and_input
        elif self.task_name == "formula":
            parse_fn = parse_context_and_question_formula
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        for item in raw_data:
            context = item.get('context', '')
            target = item.get('target', '')

            # Parse context to extract the actual text to analyze and the instruction
            input_text, question = parse_fn(context)

            processed_item = {
                "context": input_text,  # The actual context text
                "question": question,   # The instruction/question
                "target": target,       # Ground truth sentiment
                "others": {
                    "original_context": context,
                    "task": self.task_name,
                    "data_source": "finlora"
                }
            }

            processed_data.append(processed_item)

        return processed_data
    
    def _finer_answer_is_correct(self, predicted: str, ground_truth: str, return_counts=False) -> bool:
        """XBRL dataset specific answer correctness check"""
        pred = predicted.split(",")
        pred = [val.lower().strip() for val in pred]
        label = ground_truth.split(",")
        label = [val.lower().strip() for val in label]
        count = 0

        if len(pred) != len(label):
            if len(pred) > len(label):
                pred = pred[:len(label)]
            else:
                padding_needed = len(label) - len(pred)
                pred += ([""] * padding_needed)

        for prediction, ground_truth in zip(pred, label):
            try:
                ground_truth = eval(ground_truth)
                prediction = eval(prediction.replace(",", "").replace("$", ""))
            except:
                pass
            if prediction == ground_truth:
                count += 1
        score = count / len(pred) if pred else 0
        if return_counts:
            return count, len(pred)
        return score == 1 
    
    def _formula_answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """formula dataset specific answer correctness check"""
        try:
            predicted = predicted.replace(",", "")
            ground_truth = ground_truth.replace(",", "")
            return float(predicted) == float(ground_truth)
        except Exception:
            return predicted == ground_truth
        return predicted == ground_truth
    
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Dataset-specific answer correctness check.

        Args:
            predicted: Model's answer
            ground_truth: Ground truth answer

        Returns:
            bool: True if answer is correct, False otherwise
        """
        if self.task_name == "finer":
            return self._finer_answer_is_correct(predicted, ground_truth)
        elif self.task_name == "formula":
            return self._formula_answer_is_correct(predicted, ground_truth)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
    
    def _evaluate_finer_accuracy(self, out: List[str], target: List[str]) -> tuple:
        """FINER dataset specific accuracy evaluation"""
        if len(out) != len(target):
            raise ValueError("Input lists 'out' and 'target' must have the same length.")

        correct_count = 0
        total_count = 0

        for x, y in zip(out, target):
            correct, total = self._finer_answer_is_correct(x, y, return_counts=True)
            correct_count += correct
            total_count += total

        accuracy = 0.0
        if total_count > 0:
            accuracy = correct_count / total_count

        return accuracy
    
    def _evaluate_formula_accuracy(self, out: List[str], target: List[str]) -> tuple:
        """formula dataset specific accuracy evaluation"""
        if len(out) != len(target):
            raise ValueError("Input lists 'out' and 'target' must have the same length.")
        
        correct_count = 0

        for predicted, ground_truth in zip(out, target):
            if self._formula_answer_is_correct(predicted, ground_truth):
                is_correct = True
                correct_count += 1
            else:
                is_correct = False

        accuracy = 0.0
        if len(out) > 0:
            accuracy = correct_count / len(out)

        return accuracy

    
    def evaluate_accuracy(self, out: List[str], target: List[str]) -> tuple:
        """
        Dataset-specific accuracy evaluation.

        Args:
            out: List of model predictions
            target: List of ground truth targets

        Returns:
            tuple: (accuracy, response_list)
        """
        if self.task_name == "finer":
            return self._evaluate_finer_accuracy(out, target)
        elif self.task_name == "formula":
            return self._evaluate_formula_accuracy(out, target)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
    