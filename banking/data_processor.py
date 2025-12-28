"""
Data processor for banking customer support topic classification task.

This processor handles the banking dataset where:
- Input: Customer query text
- Output: Topic category from a predefined list of 77 banking topics
"""

import os
import csv
from typing import List, Dict, Any, Tuple


# All allowed topics for banking customer support classification
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


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and process data from a CSV file.
    
    Expected CSV format: text,category,group (with header)
    
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
                'category': row['category'],
                'group': row.get('group', '')
            })
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def normalize_topic(topic: str) -> str:
    """
    Normalize a topic string for comparison.
    
    Handles common variations:
    - Case insensitivity
    - Extra whitespace
    - Underscores vs spaces
    - Common typos
    """
    if not topic:
        return ""
    
    # Basic normalization
    topic = topic.strip().lower()
    
    # Replace spaces with underscores for consistency
    topic = topic.replace(' ', '_')
    
    # Remove common wrapper text that models might add
    for prefix in ['the topic is ', 'topic: ', 'category: ', 'answer: ']:
        if topic.startswith(prefix):
            topic = topic[len(prefix):]
    
    # Remove quotes if present
    topic = topic.strip('"\'')
    
    return topic


def extract_topic_from_response(response: str) -> str:
    """
    Extract the topic from a model response.
    
    The model might return the topic in various formats:
    - Just the topic name
    - "The topic is: topic_name"
    - JSON format with topic field
    - With explanation before/after
    
    Args:
        response: Raw model response
        
    Returns:
        Extracted and normalized topic
    """
    import json
    
    response = response.strip()
    
    # Try JSON parsing first
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            for key in ['topic', 'category', 'answer', 'final_answer', 'result']:
                if key in parsed:
                    return normalize_topic(str(parsed[key]))
    except (json.JSONDecodeError, KeyError):
        pass
    
    # Try to find topic patterns
    response_lower = response.lower()
    
    # Look for explicit answer patterns
    for pattern in ['final answer:', 'answer:', 'topic:', 'category:', 'the topic is']:
        if pattern in response_lower:
            idx = response_lower.find(pattern) + len(pattern)
            answer = response[idx:].strip().split('\n')[0].strip()
            answer = answer.strip('.,!?:').strip()
            return normalize_topic(answer)
    
    # If response is short (likely just the topic), use as-is
    if len(response.strip()) < 100 and '\n' not in response.strip():
        return normalize_topic(response)
    
    # Check if any allowed topic appears in the response
    normalized_topics = {normalize_topic(t): t for t in ALLOWED_TOPICS}
    
    # First, try exact match at beginning of response
    first_line = response.strip().split('\n')[0]
    first_line_normalized = normalize_topic(first_line)
    if first_line_normalized in normalized_topics:
        return first_line_normalized
    
    # Look for any topic mention
    for norm_topic, original_topic in normalized_topics.items():
        if norm_topic in response_lower.replace(' ', '_'):
            return norm_topic
    
    # Fallback: return first line normalized
    return normalize_topic(response.strip().split('\n')[0])


class DataProcessor:
    """
    Processor for handling banking customer support topic classification.
    
    This processor:
    1. Converts raw CSV data to standardized format
    2. Evaluates predictions against ground truth topics
    3. Calculates classification accuracy
    """
    
    def __init__(self, task_name: str = "banking"):
        """
        Initialize the data processor.
        
        Args:
            task_name: The name of the task (default: "banking")
        """
        self.task_name = task_name
        self.allowed_topics = ALLOWED_TOPICS
        self.topic_set = set(normalize_topic(t) for t in ALLOWED_TOPICS)
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw CSV data into standardized format for ACE.
        
        Args:
            raw_data: Raw data loaded from CSV (list of dicts with 'text', 'category')
            
        Returns:
            List of dicts with keys: 'context', 'question', 'target'
        """
        processed_data = []
        
        # Create the instruction/question that will be used for classification
        topics_list = ", ".join(self.allowed_topics)
        
        for item in raw_data:
            customer_query = item.get('text', '')
            ground_truth_topic = item.get('category', '')
            
            # The question provides the classification task instruction
            question = (
                f"Classify the following banking customer support query into one of the predefined topics.\n\n"
                f"Customer Query: {customer_query}\n\n"
                f"Available Topics: {topics_list}\n\n"
                f"Respond with ONLY the topic name, nothing else."
            )
            
            processed_item = {
                "context": "",  # No additional context needed
                "question": question,
                "target": ground_truth_topic,
                "others": {
                    "original_text": customer_query,
                    "task": self.task_name,
                }
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the predicted topic from model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Extracted topic string
        """
        return extract_topic_from_response(response)
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if the predicted topic matches the ground truth.
        
        Uses normalized comparison to handle minor variations.
        
        Args:
            predicted: Model's predicted topic
            ground_truth: Ground truth topic
            
        Returns:
            bool: True if prediction is correct, False otherwise
        """
        pred_normalized = normalize_topic(predicted)
        gt_normalized = normalize_topic(ground_truth)
        
        # Exact match after normalization
        if pred_normalized == gt_normalized:
            return True
        
        # Check if prediction contains the ground truth or vice versa
        # (handles cases like "declined_card_payment" vs "declined card payment")
        if pred_normalized in gt_normalized or gt_normalized in pred_normalized:
            return True
        
        return False
    
    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate classification accuracy across multiple predictions.
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth topics
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        if not predictions:
            return 0.0
        
        correct = sum(
            1 for pred, truth in zip(predictions, ground_truths)
            if self.answer_is_correct(pred, truth)
        )
        
        return correct / len(predictions)
    
    def get_detailed_results(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """
        Get detailed evaluation results including per-topic accuracy.
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth topics
            
        Returns:
            Dict with detailed metrics
        """
        results = {
            'total': len(predictions),
            'correct': 0,
            'accuracy': 0.0,
            'per_topic': {},
            'confusion': []
        }
        
        for pred, truth in zip(predictions, ground_truths):
            is_correct = self.answer_is_correct(pred, truth)
            
            if is_correct:
                results['correct'] += 1
            
            # Track per-topic stats
            truth_norm = normalize_topic(truth)
            if truth_norm not in results['per_topic']:
                results['per_topic'][truth_norm] = {'total': 0, 'correct': 0}
            
            results['per_topic'][truth_norm]['total'] += 1
            if is_correct:
                results['per_topic'][truth_norm]['correct'] += 1
            else:
                results['confusion'].append({
                    'ground_truth': truth,
                    'predicted': pred
                })
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        
        return results
