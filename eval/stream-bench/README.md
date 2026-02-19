# stream-bench

Benchmarking framework for text-to-SQL evaluation using the ACE (Agentic Context Engineering) system. Supports three datasets: **BIRD**, **CoSQL**, and **Spider**.


## Overview

Stream-bench evaluates the ACE system on text-to-SQL tasks using execution-based accuracy (result-set matching, not string matching). The typical workflow is:

```
Download DBs → Preprocess data → Configure task → Run ACE → Evaluate playbook → Plot results
```

ACE trains a playbook (a set of instructions) over a stream of training examples, then evaluates it on a held-out test set.


## Directory Structure

```
eval/stream-bench/
├── run.py                          # Main ACE training/evaluation runner
├── run_playbook.py                 # Evaluate a saved playbook on test data
├── plot.py                         # Generate performance plots
├── analyze_logs.py                 # Analyze terminal output logs
├── data_processor.py               # Core data loading and SQL evaluation
├── download_text2sql_data.py       # Download raw databases (BIRD/CoSQL/Spider)
├── preprocess_streambench_bird.py  # Create .jsonl from HuggingFace BIRD data
├── preprocess_streambench_cosql.py # Create .jsonl from HuggingFace CoSQL data
├── preprocess_streambench_spider.py # Create .jsonl from HuggingFace Spider data
├── dataset_stats.py                # Print dataset statistics
└── data/
    ├── bird_config.json            # Pre-defined BIRD task configurations
    ├── cosql_config.json           # Pre-defined CoSQL task configurations
    ├── spider_config.json          # Pre-defined Spider task configurations
    ├── bird/dev_databases/         # BIRD test databases (after download)
    ├── bird_train/train_databases/ # BIRD training databases (after download)
    ├── cosql/                      # CoSQL databases (after download)
    ├── spider/database/            # Spider databases (after download)
    ├── streambench_bird_train.jsonl    # Preprocessed BIRD train split
    ├── streambench_bird_val.jsonl      # Preprocessed BIRD val split
    ├── streambench_bird_test.jsonl     # Preprocessed BIRD test split
    ├── streambench_cosql_*.jsonl       # Preprocessed CoSQL splits
    └── streambench_spider_*.jsonl      # Preprocessed Spider splits
```

Results are written to a top-level `results/` directory at the repo root.


## Setup

Install dependencies from the repo root:

```bash
uv sync
# or: pip install -e .
```

Set up your API key in `.env` (copy from `.env.example`):

```bash
cp .env.example .env
# Edit .env and add your API key
```

All commands below should be run from the **repo root** (`ace/`), not from inside `eval/stream-bench/`.


## Step 1: Download Databases

The raw SQLite databases are required for execution-based evaluation. The preprocessed `.jsonl` data files (Step 2) can be downloaded from HuggingFace independently, but you still need the databases to run SQL evaluation.

### BIRD

```bash
# Test/dev databases (~350 MB)
python eval/stream-bench/download_text2sql_data.py \
    --dataset bird --split dev \
    --save_dir eval/stream-bench/data

# Training databases (~33 GB, only needed for offline mode training)
python eval/stream-bench/download_text2sql_data.py \
    --dataset bird --split train \
    --save_dir eval/stream-bench/data
```

After extraction:
```
eval/stream-bench/data/bird/dev_databases/{db_name}/{db_name}.sqlite
eval/stream-bench/data/bird_train/train_databases/{db_name}/{db_name}.sqlite
```

### CoSQL

```bash
python eval/stream-bench/download_text2sql_data.py \
    --dataset cosql \
    --save_dir eval/stream-bench/data
```

After extraction:
```
eval/stream-bench/data/cosql/{db_name}/{db_name}.sqlite
eval/stream-bench/data/cosql/tables.json
```

### Spider

```bash
python eval/stream-bench/download_text2sql_data.py \
    --dataset spider \
    --save_dir eval/stream-bench/data
```

After extraction:
```
eval/stream-bench/data/spider/database/{db_name}/{db_name}.sqlite
eval/stream-bench/data/spider/tables.json
```

> **Note:** CoSQL and Spider downloads use `gdown` and require Google Drive access. Install with `pip install gdown` if missing.


## Step 2: Preprocess Data

This step downloads the question/SQL pairs from HuggingFace (`appier-ai-research/StreamBench`) and combines them with database schema information to create `.jsonl` files used during training.

**Skip this step** if the `.jsonl` files already exist in `eval/stream-bench/data/`.

### BIRD

```bash
# Test split (uses dev databases)
python eval/stream-bench/preprocess_streambench_bird.py \
    --split test \
    --bird_root eval/stream-bench/data/bird \
    --tables_json eval/stream-bench/data/dev_20240627/dev_tables.json \
    --out eval/stream-bench/data/streambench_bird_test.jsonl

# Train split (uses train databases)
python eval/stream-bench/preprocess_streambench_bird.py \
    --split train \
    --bird_root eval/stream-bench/data/bird_train/train_databases \
    --out eval/stream-bench/data/streambench_bird_train.jsonl

# Val split
python eval/stream-bench/preprocess_streambench_bird.py \
    --split validation \
    --bird_root eval/stream-bench/data/bird_train/train_databases \
    --out eval/stream-bench/data/streambench_bird_val.jsonl
```

### CoSQL

```bash
python eval/stream-bench/preprocess_streambench_cosql.py \
    --split test \
    --cosql_root eval/stream-bench/data/cosql \
    --tables_json eval/stream-bench/data/cosql/tables.json \
    --out eval/stream-bench/data/streambench_cosql_test.jsonl

python eval/stream-bench/preprocess_streambench_cosql.py \
    --split train \
    --cosql_root eval/stream-bench/data/cosql \
    --tables_json eval/stream-bench/data/cosql/tables.json \
    --out eval/stream-bench/data/streambench_cosql_train.jsonl

python eval/stream-bench/preprocess_streambench_cosql.py \
    --split validation \
    --cosql_root eval/stream-bench/data/cosql \
    --tables_json eval/stream-bench/data/cosql/tables.json \
    --out eval/stream-bench/data/streambench_cosql_val.jsonl
```

### Spider

```bash
python eval/stream-bench/preprocess_streambench_spider.py \
    --split test \
    --spider_root eval/stream-bench/data/spider \
    --tables_json eval/stream-bench/data/spider/tables.json \
    --out eval/stream-bench/data/streambench_spider_test.jsonl

python eval/stream-bench/preprocess_streambench_spider.py \
    --split train \
    --spider_root eval/stream-bench/data/spider \
    --tables_json eval/stream-bench/data/spider/tables.json \
    --out eval/stream-bench/data/streambench_spider_train.jsonl

python eval/stream-bench/preprocess_streambench_spider.py \
    --split validation \
    --spider_root eval/stream-bench/data/spider \
    --tables_json eval/stream-bench/data/spider/tables.json \
    --out eval/stream-bench/data/streambench_spider_val.jsonl
```

Each `.jsonl` record has the format:
```json
{
  "question_id": "0",
  "question": "What is the highest eligible free rate for K-12 students in Alameda County?",
  "sql": "SELECT ...",
  "difficulty": "simple",
  "db_name": "california_schools",
  "db_schema": { "db_id": "...", "tables": [...], "primary_keys": [...], "foreign_keys": [...] }
}
```


## Step 3: Configure a Run

Runs are configured via a JSON file that specifies data paths and task parameters. Pre-built configs are in `eval/stream-bench/data/`.

### Pre-built Configurations

**BIRD** (`eval/stream-bench/data/bird_config.json`):

| Task name | Samples | Notes |
|---|---|---|
| `bird_all` | all | Full dataset |
| `bird_150` | 150 | Random subset |
| `bird_150_balanced` | 150 | Equal per difficulty |
| `bird_300_balanced` | 300 | Equal per difficulty |
| `bird_432_balanced` | 432 | Equal per difficulty |
| `bird_1000_quasi_balanced` | 1000 | Balanced with fallback |

**CoSQL** (`eval/stream-bench/data/cosql_config.json`):

| Task name | Samples | Notes |
|---|---|---|
| `cosql_all` | all | Full dataset |
| `cosql_150` | 150 | Random subset |
| `cosql_150_balanced` | 150 | Equal per difficulty |
| `cosql_36_balanced` | 36 | Small balanced subset |

**Spider** (`eval/stream-bench/data/spider_config.json`):

| Task name | Samples | Notes |
|---|---|---|
| `spider_all` | all | Full dataset |
| `spider_150` | 150 | Random subset |
| `spider_150_balanced` | 150 | Equal per difficulty |
| `spider_150_quasi_balanced` | 150 | Balanced with fallback |

### Custom Configuration

Create a JSON file with one or more task entries:

```json
{
  "my_task": {
    "train_data": "eval/stream-bench/data/streambench_bird_train.jsonl",
    "val_data":   "eval/stream-bench/data/streambench_bird_val.jsonl",
    "test_data":  "eval/stream-bench/data/streambench_bird_test.jsonl",

    "bird_train_db_root": "eval/stream-bench/data/bird_train/train_databases",
    "bird_val_db_root":   "eval/stream-bench/data/bird_train/train_databases",
    "bird_test_db_root":  "eval/stream-bench/data/bird/dev_databases",

    "max_samples": 150,
    "difficulty_filter": "quasi_balanced"
  }
}
```

For CoSQL, replace the `bird_*` keys with `cosql_db_root`:
```json
{
  "my_cosql_task": {
    "train_data": "eval/stream-bench/data/streambench_cosql_train.jsonl",
    "val_data":   "eval/stream-bench/data/streambench_cosql_val.jsonl",
    "test_data":  "eval/stream-bench/data/streambench_cosql_test.jsonl",
    "cosql_db_root": "eval/stream-bench/data/cosql",
    "max_samples": 100
  }
}
```

**Config field reference:**

| Field | Type | Description |
|---|---|---|
| `train_data` | string | Path to training `.jsonl` |
| `val_data` | string | Path to validation `.jsonl` |
| `test_data` | string | Path to test `.jsonl` |
| `bird_db_root` | string | Default database root (BIRD/Spider) |
| `bird_train_db_root` | string | Override DB root for train split |
| `bird_val_db_root` | string | Override DB root for val split |
| `bird_test_db_root` | string | Override DB root for test split |
| `cosql_db_root` | string | Database root for CoSQL tasks |
| `max_samples` | int | Cap on samples (applies to all splits) |
| `max_train_samples` | int | Cap for train split only |
| `max_val_samples` | int | Cap for val split only |
| `max_test_samples` | int | Cap for test split only |
| `difficulty_filter` | string | `simple-only`, `moderate-only`, `challenging-only`, `balanced`, `quasi_balanced` |


## Step 4: Run ACE Training

```bash
python eval/stream-bench/run.py \
    --data_config eval/stream-bench/data/bird_config.json \
    --task_name bird_150_balanced \
    --mode online \
    --api_provider sambanova \
    --generator_model DeepSeek-V3-0324 \
    --curator_model DeepSeek-V3-0324 \
    --reflector_model DeepSeek-V3-0324
```

### Run modes

| Mode | Description |
|---|---|
| `online` | Stream training data one-by-one, updating the playbook after each window |
| `offline` | Train on the full training set, then evaluate on test |
| `eval_only` | Skip training, evaluate test set with an existing or empty playbook |

### Commonly used options

| Flag | Default | Description |
|---|---|---|
| `--mode` | required | `online`, `offline`, or `eval_only` |
| `--task_name` | required | Key from the config JSON |
| `--data_config` | required | Path to config JSON |
| `--api_provider` | required | `sambanova`, `together`, or `openai` |
| `--generator_model` | required | Model name for SQL generation |
| `--curriculum` | none | `easy_to_hard`, `hard_to_easy`, or `random` |
| `--num_epochs` | 1 | Number of passes over training data |
| `--max_num_rounds` | — | Maximum curator/reflector rounds |
| `--eval_steps` | — | Evaluate every N training steps |
| `--playbook_token_budget` | — | Token budget for playbook size |
| `--test_workers` | 1 | Parallel workers for test evaluation |
| `--initial_playbook_path` | none | Warm-start from an existing playbook |
| `--plot` | false | Auto-generate plots after run completes |


## Step 5: Evaluate a Playbook

After a run completes, evaluate a specific saved playbook on any data split:

```bash
python eval/stream-bench/run_playbook.py \
    --results_dir results/ace_run_20260119_234301_bird_150_balanced_online \
    --playbook_file intermediate_playbooks/window_4_final_playbook.txt \
    --dataset test
```

To run a baseline evaluation with an empty playbook:

```bash
python eval/stream-bench/run_playbook.py \
    --results_dir results/ace_run_20260119_234301_bird_150_balanced_online \
    --dataset test
```

Save detailed per-sample results to a JSON file:

```bash
python eval/stream-bench/run_playbook.py \
    --results_dir results/ace_run_20260119_234301_bird_150_balanced_online \
    --playbook_file intermediate_playbooks/window_4_final_playbook.txt \
    --output_file playbook_eval_results.json \
    --dataset test
```

> **Note:** `run_playbook.py` reads samples from the `processed_data/` subfolder that `run.py` writes during its setup phase. Run `run.py` at least once first so that folder exists.


## Results Structure

Each run creates a timestamped directory under `results/`:

```
results/
└── ace_run_YYYYMMDD_HHMMSS_{task_name}_{mode}/
    ├── run_config.json                    # Full run configuration snapshot
    ├── final_playbook.txt                 # Playbook after all training
    ├── intermediate_playbooks/
    │   ├── window_1_final_playbook.txt
    │   ├── window_2_final_playbook.txt
    │   └── ...
    ├── processed_data/
    │   ├── train_samples.json             # Preprocessed training data
    │   ├── val_samples.json               # Preprocessed validation data
    │   └── test_samples.json              # Preprocessed test data
    └── terminal_output.log                # Full terminal log
```

Each `*_samples.json` is a list of objects with:
```json
{
  "context": "<schema + instructions passed to the model>",
  "question": "natural language question",
  "target": "ground truth SQL",
  "others": {
    "question_id": "0",
    "difficulty": "moderate",
    "db_name": "california_schools",
    "task": "bird",
    "data_source": "streambench_bird"
  }
}
```


## Plotting

Generate accuracy-over-time plots for a completed run:

```bash
# Online run
python eval/stream-bench/plot.py \
    --run_dir results/ace_run_20260116_103642_bird_150_balanced_online \
    --mode online

# Offline run
python eval/stream-bench/plot.py \
    --run_dir results/ace_run_20260122_203526_bird_150_balanced_offline \
    --mode offline
```

Or pass `--plot` to `run.py` to generate plots automatically at the end of a run.


## Reference: All CLI Arguments

### run.py

```
Required:
  --data_config PATH            Config JSON file
  --task_name TEXT              Task key within the config JSON
  --mode {online|offline|eval_only}
  --api_provider {sambanova|together|openai}
  --generator_model TEXT

Optional (model):
  --reflector_model TEXT
  --curator_model TEXT
  --max_tokens INT

Optional (training):
  --curriculum {easy_to_hard|hard_to_easy|random}
  --num_epochs INT
  --max_num_rounds INT
  --curator_frequency INT
  --eval_steps INT
  --online_eval_frequency INT
  --save_steps INT
  --playbook_token_budget INT
  --initial_playbook_path PATH
  --pass_sql_eval_results / --no_pass_sql_eval_results
  --json_mode / --no_json_mode
  --no_ground_truth
  --use_bulletpoint_analyzer / --no_bulletpoint_analyzer
  --bulletpoint_analyzer_threshold INT

Optional (output):
  --save_path PATH
  --test_workers INT
  --plot
```

### run_playbook.py

```
Required:
  --results_dir PATH            ACE results directory

Optional:
  --playbook_file PATH          Playbook file relative to results_dir
                                (omit for empty-playbook baseline)
  --dataset {train|val|test}    Split to evaluate (default: test)
  --bird_db_root PATH           BIRD database root (auto-detected by dataset)
  --api_provider {sambanova|together|openai}
  --generator_model TEXT
  --num_workers INT             Parallel generation workers (default: 4)
  --output_file PATH            Save detailed per-sample results as JSON
```

### download_text2sql_data.py

```
  --dataset {bird|cosql|spider}
  --split {dev|train}           BIRD only (default: dev)
  --save_dir PATH               Where to save files (default: ./data)
```

### preprocess_streambench_{bird,cosql,spider}.py

```
  --split {train|validation|test}
  --out PATH                    Output .jsonl file
  --tables_json PATH            Path to tables.json (recommended for schema)
  --bird_root / --cosql_root / --spider_root PATH   DB root (fallback schema source)
  --schema_format {json|string} (default: json)
  --train_ratio FLOAT           Train/val split ratio (default: 0.8, BIRD/Spider only)
  --seed INT                    Random seed (default: 42)
```
