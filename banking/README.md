# Banking Customer Support Topic Classification

This directory contains a complete solution for classifying banking customer support queries into one of **77 predefined topics** using prompt optimization techniques.

## 📋 Task Overview

The task is to classify customer support messages (e.g., "I'm not sure why my card didn't work") into specific banking categories (e.g., `declined_card_payment`).

### Dataset

The dataset is split into three parts:
- **Training set** (`data/train.csv`): 250 samples for learning
- **Validation set** (`data/val.csv`): Samples for hyperparameter tuning
- **Test set** (`data/test.csv`): Held-out samples for final evaluation

Each sample contains:
- `text`: Customer query (e.g., "My card still hasn't been delivered")
- `category`: Ground truth topic (e.g., `card_arrival`)
- `group`: Data split identifier

### Topic Categories (77 total)

The classifier must choose from categories including:
- Card-related: `card_arrival`, `card_not_working`, `declined_card_payment`, `lost_or_stolen_card`, etc.
- Transfer-related: `failed_transfer`, `pending_transfer`, `declined_transfer`, etc.
- Top-up related: `top_up_failed`, `pending_top_up`, `top_up_reverted`, etc.
- Account/Identity: `verify_my_identity`, `edit_personal_details`, `terminate_account`, etc.
- Fees/Charges: `card_payment_fee_charged`, `exchange_charge`, `cash_withdrawal_charge`, etc.

## 🏗️ Solution Architecture

This solution implements two prompt optimization approaches:

### 1. ACE (Agentic Context Engineering)

ACE uses a three-component architecture:
- **Generator**: Classifies customer queries using the current playbook
- **Reflector**: Analyzes errors and generates improvement suggestions
- **Curator**: Maintains and updates the playbook with learned rules

The playbook contains structured rules organized by:
- Classification principles
- Category disambiguation rules
- Banking domain knowledge
- Common patterns
- Common mistakes to avoid

### 2. GEPA (Gradient-free Efficient Prompt Adaptation)

GEPA is a DSPy optimizer that improves prompts through reflection:
- **ChainOfThought**: Base reasoning module for classification
- **Reflection-based optimization**: Analyzes errors and generates feedback to improve instructions
- **Metric with feedback**: Custom evaluation that provides detailed guidance on misclassifications

## 📁 File Structure

```
banking/
├── README.md                    # This file
├── __init__.py                  # Package initializer
├── data_processor.py            # Data loading and evaluation logic
├── run.py                       # Main ACE training script
├── run_ace_workflow.py          # Complete ACE workflow (baseline → train → eval)
├── run_gepa_workflow.py         # Complete GEPA workflow (baseline → train → eval)
├── data/
│   ├── task_config.json         # Data path configuration
│   ├── train.csv                # Training data
│   ├── val.csv                  # Validation data
│   └── test.csv                 # Test data
└── results/                     # Experiment outputs
    ├── banking_*/               # ACE run results
    └── gepa_*/                  # GEPA run results
```

## 🚀 Usage

### Prerequisites

```bash
# Set up API key
export ANTHROPIC_API_KEY="your-api-key"

# Navigate to the ace directory
cd /path/to/prompt_optimisation_gepa_ace/ace
```

### Running ACE Workflow

```bash
# Full workflow: baseline → training → final evaluation
python -m banking.run_ace_workflow

# Quick test with limited samples
python -m banking.run_ace_workflow --max-train 20 --max-test 30

# Skip baseline (if already evaluated)
python -m banking.run_ace_workflow --skip-baseline

# Use existing playbook for evaluation only
python -m banking.run_ace_workflow --skip-training --playbook results/banking_*/best_playbook.txt
```

### Running GEPA Workflow

```bash
# Full GEPA workflow
python -m banking.run_gepa_workflow

# Quick test with limited samples
python -m banking.run_gepa_workflow --max-train 20 --max-test 30

# Skip baseline evaluation
python -m banking.run_gepa_workflow --skip-baseline
```

### Using the Low-Level ACE API

```bash
# Offline training with validation
python -m banking.run \
    --task_name banking \
    --mode offline \
    --save_path results/my_experiment

# Online training (train and test on same data)
python -m banking.run \
    --task_name banking \
    --mode online \
    --save_path results/my_experiment

# Evaluation only with pre-trained playbook
python -m banking.run \
    --task_name banking \
    --mode eval_only \
    --initial_playbook_path results/banking/best_playbook.txt \
    --save_path results/my_eval
```

## ⚙️ Configuration Options

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--api_provider` | `anthropic` | API provider (anthropic, openai, together, sambanova) |
| `--generator_model` | `claude-3-5-haiku-20241022` | Model for classification |
| `--reflector_model` | `claude-3-5-haiku-20241022` | Model for error analysis |
| `--curator_model` | `claude-3-5-haiku-20241022` | Model for playbook curation |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | 1 | Number of training epochs |
| `--max_num_rounds` | 3 | Max reflection rounds per error |
| `--curator_frequency` | 1 | Run curator every N steps |
| `--eval_steps` | 50 | Evaluate every N steps |
| `--save_steps` | 25 | Save intermediate playbooks every N steps |

## 🧠 Key Components

### DataProcessor (`data_processor.py`)

Handles data loading and evaluation:
- `load_data()`: Loads CSV files into structured format
- `normalize_topic()`: Normalizes topic strings for comparison
- `extract_topic_from_response()`: Extracts topic from LLM responses
- `DataProcessor.evaluate()`: Computes classification accuracy

### ACE Playbook

The learned playbook contains structured rules like:

```
[cls-00008] helpful=5 harmful=1 :: Match the transaction type mentioned 
in the query (top-up, transfer, payment, card, etc.) to categories with 
the same transaction type prefix.

[dis-00006] helpful=1 harmful=0 :: card_not_working vs declined_card_payment: 
Use 'declined_card_payment' when the query suggests a transaction attempt 
that failed. Use 'card_not_working' only for physical/technical card issues.

[pat-00005] helpful=1 harmful=0 :: When customers say 'my card didn't work' 
without additional context, this typically means a payment was declined, 
not a physical malfunction.
```

Each rule has:
- **ID**: Unique identifier
- **helpful/harmful scores**: Track rule effectiveness
- **Rule text**: The actual classification guidance

## 📝 Output Files

After running, results are saved in timestamped directories:

```
results/banking_YYYYMMDD_HHMMSS/
├── REPORT.md              # Human-readable summary
├── summary.json           # Machine-readable results
├── best_playbook.txt      # Best performing playbook
├── baseline/              # Baseline evaluation results
├── training/              # Training logs and intermediate playbooks
│   ├── bullet_usage_log.jsonl
│   ├── curator_operations_diff.jsonl
│   ├── intermediate_playbooks/
│   └── detailed_llm_logs/
└── final/                 # Final evaluation results
```

## 🔧 Extending the Solution

### Adding New Categories

1. Update `ALLOWED_TOPICS` in `data_processor.py`
2. Add training examples to `data/train.csv`
3. Re-run training

### Custom Evaluation Metrics

Override the `evaluate()` method in `DataProcessor`:

```python
def evaluate(self, predictions, ground_truth):
    # Custom evaluation logic
    pass
```

### Using Different Models

```bash
python -m banking.run_ace_workflow \
    --generator-model gpt-4o \
    --reflector-model gpt-4o \
    --curator-model gpt-4o \
    --api-provider openai
```