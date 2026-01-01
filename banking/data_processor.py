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
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if the predicted topic matches the ground truth.
        
        Uses simple case-insensitive comparison for consistency with GEPA metric.
        
        Args:
            predicted: Model's predicted topic
            ground_truth: Ground truth topic
            
        Returns:
            bool: True if prediction is correct, False otherwise
        """
        return predicted.lower() == ground_truth.lower()
    
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
