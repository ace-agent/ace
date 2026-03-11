# 🛠️ Extending ACE: Developer Guide

This guide provides detailed instructions for adding new tasks to the ACE framework.

## 📁 Repository Structure

Understanding the codebase structure will help you navigate and extend ACE effectively:

```
ACE-pre-release/
├── ace/                         # Core ACE framework
│   ├── core/                    # Agent implementations
│   │   ├── __init__.py
│   │   ├── generator.py         # Generator agent
│   │   ├── reflector.py         # Reflector agent
│   │   ├── curator.py           # Curator agent
│   │   └── bulletpoint_analyzer.py       # Bulletpoint analyzer for playbook de-duplication
│   ├── prompts/                 # Prompt templates
│   │   ├── __init__.py
│   │   ├── generator.py         # Generator prompts
│   │   ├── reflector.py         # Reflector prompts
│   │   └── curator.py           # Curator prompts
│   ├── __init__.py
│   └── ace.py                   # Main ACE orchestrator
│
├── finance/                     # Finance domain implementation (reference example)
│   ├── data_processor.py        # Finance data processing
│   └── run.py                   # Unified training and evaluation script
│
├── llm.py                       # LLM utilities
├── logger.py                    # Logging utilities
├── utils.py                     # General utilities 
├── playbook_utils.py            # Playbook operations
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── README.md                    # Main documentation
└── EXTENDING_ACE.md             # This file
```

## Adding New Tasks

To add a new task to ACE, follow these four steps:

### Step 1: Prepare Your Data

Create JSONL files for train/validation/test splits. Each line should be a JSON object. The field names can be anything - your `process_task_data()` method will handle the mapping. For example:

```json
{"context": "your prompt/instruction text", "target": "ground truth answer"}
```

Or with custom field names:
```json
{"input": "question text", "output": "answer", "metadata": {...}}
```

Create a configuration file (e.g., `your_task/data/task_config.json`):
```json
{
    "your_task_name": {
        "train_data": "./your_task/data/train.jsonl",
        "val_data": "./your_task/data/val.jsonl",
        "test_data": "./your_task/data/test.jsonl"
    }
}
```

### Step 2: Create a Data Processor

Create `your_task/data_processor.py` with a `DataProcessor` class. You only need to implement **3 simple methods**:

```python
# your_task/data_processor.py
import os
import json
from typing import List, Dict, Any, Tuple


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    """
    Processor for handling data preprocessing and evaluation.
    
    You only need to implement 3 methods:
    1. process_task_data() - Convert raw data to standardized format
    2. answer_is_correct() - Check if a prediction matches ground truth
    3. evaluate_accuracy() - Calculate overall accuracy
    
    The evaluation orchestration is handled by utils.evaluate_test_set().
    """
    
    def __init__(self, task_name: str):
        """Initialize with task name."""
        self.task_name = task_name
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw data into standardized format.
        
        Args:
            raw_data: Raw data loaded from JSONL
            
        Returns:
            List of dicts with keys: 'context', 'question', 'target'
        """
        processed_data = []
        
        for item in raw_data:
            # Apply any task-specific preprocess here
            context, question, target = self._prepare_input(item)
            
            processed_item = {
                "context": context,      # Background information
                "question": question,    # The actual question/instruction
                "target": target,        # Ground truth answer
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def _prepare_input(self, item: dict) -> Tuple[str, str, str]:
        """
        Extract and parse data fields into (context, question, target).
        Customize this helper method for your task's data format.
        """
        context = item.get('context', '')
        question = item.get('question', '')
        target = item.get('target', '')
        return context, question, target
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        Implement task-specific comparison logic.
        
        This is called by the evaluation utilities in utils.py.
        """
        # Example: exact match (case-insensitive)
        return predicted.strip().lower() == ground_truth.strip().lower()
        
        # Or numeric comparison:
        # try:
        #     return float(predicted) == float(ground_truth)
        # except:
        #     return predicted == ground_truth
    
    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate accuracy across multiple predictions.
        
        This is called by the evaluation utilities in utils.py.
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        correct = sum(
            1 for pred, truth in zip(predictions, ground_truths)
            if self.answer_is_correct(pred, truth)
        )
        
        return correct / len(predictions) if predictions else 0.0
```

### Step 3: Create a Training Script

Create `your_task/run.py`:

```python
#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
from .data_processor import DataProcessor, load_data

from ace import ACE
from utils import initialize_clients


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ACE on your task')
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default="offline",
                        choices=['offline', 'online', 'eval_only'],
                        help="Run mode: 'offline' for offline training, "
                             "'online' for online training, 'eval_only' for evaluation only")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--initial_playbook_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="./your_task/data/task_config.json")
    # Add other arguments as needed (see finance/run.py for full list)
    return parser.parse_args()


def preprocess_data(task_name, config, mode):
    """Load and preprocess data."""
    processor = DataProcessor(task_name=task_name)
    
    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")
        
        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")
    
    # For offline mode, load train, val, and optionally test data
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []
        
        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")
    
    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def main():
    args = parse_args()
    
    # Load task configuration
    with open(args.config_path, 'r') as f:
        task_config = json.load(f)
    
    # Preprocess data
    train_samples, val_samples, test_samples, data_processor = \
        preprocess_data(args.task_name, task_config[args.task_name], args.mode)
    
    # Load initial playbook (or use empty if None provided)
    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")
    
    # Initialize ACE
    api_provider = "sambanova" # or "together", "openai", "commonstack"
    ace_system = ACE(
        api_provider=api_provider,
        generator_model="DeepSeek-V3.1",  # Or your preferred model
        reflector_model="DeepSeek-V3.1",
        curator_model="DeepSeek-V3.1",
        max_tokens=4096,
        initial_playbook=initial_playbook
    )
    
    # Configure
    config = {
        'num_epochs': 1,
        'max_num_rounds': 3,
        'curator_frequency': 1,
        'eval_steps': 100,
        'online_eval_frequency': 15,
        'save_steps': 50,
        'playbook_token_budget': 80000,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': False,
        'no_ground_truth': False,
        'save_dir': args.save_path,
        'test_workers': 20,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': false,   # Turn on for playbook bulletpoints de-duplication and merging
        'api_provider': api_provider
    }
    
    # Run using the unified interface
    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config
    )
   

if __name__ == "__main__":
    main()
```

### Step 4: Run Training

```bash
# Offline training (with automatic initial and final testing)
python -m your_task.run \
    --task_name your_task_name \
    --mode offline \
    --save_path results \
    --config_path ./your_task/data/task_config.json

# Online training and testing
python -m your_task.run \
    --task_name your_task_name \
    --mode online \
    --save_path results \
    --config_path ./your_task/data/task_config.json

# Evaluation only (test a pre-trained playbook)
python -m your_task.run \
    --task_name your_task_name \
    --mode eval_only \
    --initial_playbook_path results/ace_run_timestamp/best_playbook.txt \
    --save_path test_results \
    --config_path ./your_task/data/task_config.json
```

## Key Implementation Notes

### 1. DataProcessor Interface

Your `DataProcessor` class only needs to implement **3 methods**:

- **`process_task_data(raw_data)`**: Convert raw JSONL data to standardized format with keys `context`, `question`, `target`
- **`answer_is_correct(predicted, ground_truth)`**: Task-specific comparison logic (exact match, numeric comparison, fuzzy matching, etc.)
- **`evaluate_accuracy(predictions, ground_truths)`**: Calculate overall accuracy metric

### 2. Evaluation Utilities

The parallel evaluation logic is handled by reusable functions in `utils.py`:

- **`evaluate_single_test_sample()`**: Evaluates a single sample (used internally)
- **`evaluate_test_set()`**: Orchestrates parallel evaluation across the test set

You don't need to implement these - they work automatically with any `DataProcessor` that implements the 3 required methods.

### 3. Data Format

The `process_task_data` method must return a list of dictionaries with these exact keys:

- **`context`**: Background information or input text
- **`question`**: The question or instruction
- **`target`**: Ground truth answer

### 4. Answer Checking

Implement `answer_is_correct()` with logic appropriate for your task:

- Exact match (case-sensitive or insensitive)
- Numeric comparison with tolerance
- Structural matching (e.g., for lists or JSON)
- Custom domain-specific logic

### 5. Model Selection

You can use any OpenAI-compatible model by changing the model names in the training script. The framework has been tested with various models including GPT-4 and open-source models like DeepSeek and Llama.

## Customizing Prompts

To adapt ACE's prompts to your domain, modify the prompt templates in `ace/prompts/`:

```python
# ace/prompts/generator.py
# Customize the generator system prompt for your domain

# ace/prompts/reflector.py  
# Customize the reflector's evaluation criteria

# ace/prompts/curator.py
# Customize how insights are curated into the playbook
```

### Example: Domain-Specific Generator Prompt

```python
# In ace/prompts/generator.py

MEDICAL_GENERATOR_PROMPT = """
You are a medical AI assistant specializing in clinical decision support.
When answering questions:
1. Always prioritize patient safety
2. Cite medical evidence when available
3. Acknowledge uncertainty when appropriate
4. Consider differential diagnoses

{playbook}

Question: {question}
Context: {context}
"""
```

## Reference Implementation

The `finance/` directory contains a complete working example of a custom task implementation. Use it as a reference for:

- Data preprocessing with multiple parsing strategies (`parse_instruction_and_input`, `parse_context_and_question_formula`)
- Task-specific evaluation logic (`_finer_answer_is_correct`, `_formula_answer_is_correct`)
- Handling different data formats and answer types
- Using the unified `run()` interface with different modes

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure your task directory has an `__init__.py` file
2. **Data format mismatches**: Verify your `process_task_data` returns the correct dictionary structure
3. **Evaluation errors**: Check that `answer_is_correct` handles edge cases (empty strings, None values, etc.)
4. **Memory issues**: Reduce `test_workers` parameter if running into memory constraints

### Getting Help

- **Issues**: Open an issue on GitHub with details about your task and error messages
- **Discussions**: Join the [GitHub Discussions](../../discussions) for implementation questions
- **Examples**: Check the `finance/` directory for working reference implementations


---

For more examples and updates, check the [main README](README.md) and [arXiv paper](https://arxiv.org/abs/2510.04618).
