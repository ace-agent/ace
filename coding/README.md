# Python Code Generation Task (MBPP Dataset)

This directory contains the implementation for the **Python code generation task** using the **MBPP (Mostly Basic Python Problems)** dataset with ACE and GEPA prompt optimization frameworks.

## 📊 Task Overview

### Dataset: MBPP (Mostly Basic Python Problems)

- **Input**: Natural language problem description + test cases
- **Output**: Python code solution
- **Evaluation**: Code execution - all test cases must pass
- **Metric**: Binary accuracy (1 if all tests pass, 0 otherwise)

### Dataset Format

Each sample contains:
- `text`: Problem description in natural language
- `code`: Ground truth Python solution (reference only)
- `test_list`: List of assertion statements to validate the code
- `group`: Optional grouping field

**Example**:
```csv
text,code,test_list,group
"Write a function to check if a triangle is scalene","def check_scalene(x,y,z):
    if x!=y and y!=z and z!=x:
        return True
    else:
        return False","['assert check_scalene(6,8,12)==True', 'assert check_scalene(6,6,12)==False']",basic
```

## 🏗️ Architecture

### Data Processor (`data_processor.py`)

Handles code generation-specific operations:

1. **Data Loading**: Loads CSV data with problem descriptions and test cases
2. **Code Extraction**: Extracts Python code from model responses (handles markdown, JSON, etc.)
3. **Code Execution**: Safely executes generated code with test assertions
4. **Evaluation**: Determines if all test cases pass

**Key Features**:
- Timeout protection (5 seconds per test)
- Isolated execution environment
- Detailed error reporting (syntax errors, runtime errors, test failures)
- Comprehensive test result tracking

### Workflows

#### 1. ACE Workflow (`run_ace_workflow.py`)

Uses the ACE (Automatic Curriculum Expert) framework:

**Playbook Sections**:
```
## GENERAL
## CODE GENERATION PRINCIPLES
## COMMON PYTHON PATTERNS
## HANDLING EDGE CASES
## ALGORITHM DESIGN
## TEST CASE INTERPRETATION
## COMMON MISTAKES TO AVOID
## DEBUGGING STRATEGIES
## OTHERS
```

**Steps**:
1. **Baseline Evaluation**: Test without playbook
2. **ACE Training**: Learn best practices through reflection and curation
3. **Final Evaluation**: Test with optimized playbook

#### 2. GEPA Workflow (`run_gepa_workflow.py`)

Uses the GEPA (Gradient-based Evolution of Prompts with Adaptation) framework:

**DSPy Signature**:
```python
class CodeResponse(dspy.Signature):
    """You are an expert Python programmer. Given a problem description 
    and test cases, write a Python function that solves the problem and 
    passes all tests."""
    
    problem = dspy.InputField(desc='Problem description in natural language')
    test_cases = dspy.InputField(desc='Test assertions the code must pass')
    code: str = dspy.OutputField(desc='Python code solution')
```

**Feedback Types**:
- ✓ All tests pass
- ✗ Timeout (infinite loops, inefficient algorithms)
- ✗ Syntax errors
- ✗ All tests fail (fundamental logic issues)
- ✗ Partial success (missing edge cases)

#### 3. Unified Runner (`run.py`)

Provides flexible training modes:
- **Offline**: Train on train set, validate on val set, test on test set
- **Online**: Train and test on test set
- **Eval-only**: Test with pre-trained playbook

## 🚀 Usage

### Prerequisites

```bash
# Ensure you're in the ace directory
cd /path/to/prompt_optimisation_gepa_ace/ace

# Activate virtual environment
source ../.venv/bin/activate

# Set API key
export ANTHROPIC_API_KEY="your-api-key"
```

### Quick Start - ACE Workflow

```bash
# Full workflow with small sample (for testing)
python -m coding.run_ace_workflow \
    --max-train 20 \
    --max-test 30

# Skip baseline evaluation
python -m coding.run_ace_workflow \
    --skip-baseline \
    --max-train 50 \
    --max-test 100

# Use existing playbook for evaluation
python -m coding.run_ace_workflow \
    --skip-training \
    --playbook results/coding_20231215_103045/best_playbook.txt \
    --max-test 100
```

### Quick Start - GEPA Workflow

```bash
# Full workflow with small sample
python -m coding.run_gepa_workflow \
    --max-train 20 \
    --max-test 30

# Adjust optimization intensity
python -m coding.run_gepa_workflow \
    --auto-mode heavy \
    --max-train 100 \
    --max-test 50

# Skip baseline evaluation
python -m coding.run_gepa_workflow \
    --skip-baseline \
    --max-train 50 \
    --max-test 100
```

### Unified Runner

```bash
# Offline training (recommended)
python -m coding.run \
    --task_name coding \
    --mode offline \
    --save_path results/coding_offline

# Online training
python -m coding.run \
    --task_name coding \
    --mode online \
    --save_path results/coding_online

# Evaluation only
python -m coding.run \
    --task_name coding \
    --mode eval_only \
    --initial_playbook_path results/coding_offline/best_playbook.txt \
    --save_path results/coding_eval
```

## ⚙️ Configuration Options

### ACE Workflow Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-train` | None | Limit training samples |
| `--max-test` | None | Limit test samples |
| `--api-provider` | anthropic | API provider (anthropic, openai, together, sambanova) |
| `--generator-model` | claude-haiku-4-5 | Model for code generation |
| `--reflector-model` | claude-sonnet-4-5 | Model for reflection |
| `--curator-model` | claude-sonnet-4-5 | Model for curation |
| `--skip-baseline` | False | Skip baseline evaluation |
| `--skip-training` | False | Skip ACE training |
| `--playbook` | None | Path to existing playbook |

### GEPA Workflow Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-train` | None | Limit training samples |
| `--max-test` | None | Limit test samples |
| `--main-model` | claude-haiku-4-5 | Main LM for generation |
| `--reflection-model` | claude-sonnet-4-20250514 | Reflection LM for optimization |
| `--num-threads` | 16 | Number of parallel threads |
| `--auto-mode` | medium | Optimization intensity (light, medium, heavy) |
| `--skip-baseline` | False | Skip baseline evaluation |
| `--skip-training` | False | Skip GEPA training |

## 📁 File Structure

```
coding/
├── README.md                   # This file
├── __init__.py                # Package initialization
├── data_processor.py          # Code execution and evaluation
├── run_ace_workflow.py        # ACE prompt optimization
├── run_gepa_workflow.py       # GEPA prompt optimization
├── run.py                     # Unified runner script
├── data/
│   ├── task_config.json       # Data paths configuration
│   ├── train.csv              # Training data
│   ├── val.csv                # Validation data
│   └── test.csv               # Test data
└── results/                   # Generated results (created at runtime)
    └── coding_YYYYMMDD_HHMMSS/
        ├── REPORT.md          # Human-readable report
        ├── summary.json       # Results summary
        ├── best_playbook.txt  # Optimized playbook (ACE)
        ├── optimized_instructions.txt  # Optimized instructions (GEPA)
        ├── baseline/          # Baseline evaluation results
        ├── training/          # Training artifacts
        └── final/             # Final evaluation results
```

## 🔍 Key Differences from Banking Task

| Aspect | Banking Task | Coding Task |
|--------|--------------|-------------|
| **Output Type** | Fixed category (77 options) | Free-form Python code |
| **Validation** | String comparison | Code execution with tests |
| **Error Feedback** | Wrong category | Syntax/runtime/test errors |
| **Complexity** | Single classification | Multi-step code generation |
| **Safety** | None needed | Timeout & sandboxing required |
| **Metric Granularity** | Binary (correct/wrong) | Can track partial (X of Y tests) |

## 🔒 Safety Measures

Code execution includes multiple safety mechanisms:

1. **Timeout**: 5-second limit per test to prevent infinite loops
2. **Isolated Environment**: Code runs in controlled namespace
3. **Error Handling**: Catches and reports syntax, runtime, and assertion errors
4. **Signal-based Interrupts**: Uses SIGALRM for reliable timeout enforcement

## 📊 Expected Results

Based on MBPP dataset characteristics:

- **Baseline Accuracy**: 20-40% (varies by model)
- **Post-Optimization Accuracy**: 30-50% (varies by dataset size and model)
- **Common Failure Modes**:
  - Missing edge cases (empty input, None, negative numbers)
  - Off-by-one errors in loops
  - Incorrect function signatures
  - Type conversion issues

## 🐛 Troubleshooting

### Issue: "Code execution timed out"

**Solution**: The generated code has an infinite loop or is too slow. The model needs to learn more efficient algorithms.

### Issue: "Syntax error in generated code"

**Solution**: The model is generating malformed Python. Ensure the playbook emphasizes proper syntax and structure.

### Issue: "Function name doesn't match tests"

**Solution**: Model needs to carefully read test cases to infer the correct function name and signature.

### Issue: "All tests fail immediately"

**Solution**: Check that the test cases are being passed correctly to the model. Review the question formatting in `data_processor.py`.

## 📚 Additional Resources

- **MBPP Dataset**: [GitHub - google-research/mbpp](https://github.com/google-research/mbpp)
- **ACE Paper**: See main ACE repository documentation
- **GEPA/DSPy**: [DSPy Documentation](https://dspy-docs.vercel.app/)

## 🤝 Contributing

When adding new features:

1. Maintain compatibility with the banking task structure
2. Update `MIGRATION_PLAN.md` if changing core interfaces
3. Test with `--max-train 10 --max-test 10` before full runs
4. Document any new safety measures for code execution

## 📝 Notes

- The data processor handles various code formatting (markdown, JSON, plain text)
- Test execution is sequential to avoid race conditions
- Ground truth code is reference only - evaluation is purely test-based
- Consider increasing timeout for more complex problems
