# Terminal-Bench / ComBEE reproduction contract

This guide documents the repository's native prompt and operation interfaces in
response to [issue #41](https://github.com/ace-agent/ace/issues/41). It does **not**
provide the paper's Terminal-Bench prompts, a terminal execution harness, or a
reproduction of benchmark scores. Issue #41 reports a 60-train/29-test experiment;
those counts alone do not identify its task split or configuration.

## Run the offline role contracts

From the repository root, in the project's installed environment:

```sh
python -m unittest discover -s tests -p 'test_combee_prompt_contract.py' -v
```

For a smaller isolated environment used only for these contracts:

```sh
python3.12 -m venv .venv-contract
.venv-contract/bin/python -m pip install openai tiktoken python-dotenv numpy
.venv-contract/bin/python -m unittest discover -s tests -p 'test_combee_prompt_contract.py' -v
```

These are the import dependencies of this test path, not a replacement for the
project's full dependencies. The optional sentence-transformers/faiss warning is
expected in this smaller environment: no deduplication code is exercised. No API
key or model download is needed. Eight tests import the real roles and templates,
replace only `timed_llm_call` with deterministic JSON, and exercise native prompt
formatting, validation, filtering and playbook application. Socket connections are
rejected during the tests. Logs use temporary directories.

The tests do not exercise ACEBatch scheduling, terminal command execution,
model quality, retries, embedding deduplication or benchmark accuracy. In the
reducer test the caller explicitly applies the returned operations once; this is
not an end-to-end assertion about the ACEBatch runner.

## Record prompt provenance

Baseline: `82709de050e1db6e6ef2f07bcb0393560b94992a`. The following are SHA-256
digests of **unformatted template strings encoded as UTF-8**, including whitespace,
not digests of Python files or populated prompts. These are repository defaults,
not verified official Terminal-Bench experiment prompts.

| Source / constant | SHA-256 |
|---|---|
| `ace/prompts/reflector.py`: `REFLECTOR_PROMPT` | `b7d29c97e755ad1c676e8252bc153c38c71d651b4432232029922b8b1e17742c` |
| same: `REFLECTOR_PROMPT_NO_GT` | `4607c858da1efdff520511a11808c9a77910122e19ab4709ad7b4b50c63d33f5` |
| `ace/prompts/curator.py`: `CURATOR_PROMPT` | `d6c19108b180badc7e02f9fe44decc3d076fb5e7c9b7299bfcc8b97a20e207b0` |
| same: `CURATOR_PROMPT_NO_GT` | `012ca075fc347364d47589f5f3bc09fd357222d8851a0206faa4e4f19a18a77a` |
| same: `CURATOR_OPERATIONS_AGGREGATION_PROMPT` | `15c17b922f42bcf687a18bdae14c94f4c935a2eea16fbff42ea289d8ae2c51f4` |

Recompute after a checkout or intentional template edit:

```sh
python - <<'PY'
import hashlib
import runpy
for path in ('ace/prompts/reflector.py', 'ace/prompts/curator.py'):
    for name, value in runpy.run_path(path).items():
        if name.isupper() and isinstance(value, str):
            print(path, name, hashlib.sha256(value.encode('utf-8')).hexdigest())
PY
```

Record the generator prompt/harness separately, plus hashes of any custom
templates actually used. Pin model identifiers, decoding parameters, provider,
ACE commit, Python/dependency versions, harness/agent and container-image
versions, task IDs and dataset version. Keep train/test IDs disjoint. Capture
seeds, task order, retries, timeouts and per-task outcomes; totals are insufficient.

## Map terminal evidence to native role inputs

An external harness must define and record this mapping; this repository does not
automatically run Terminal-Bench tasks or collect their terminal observations.

| Native field | Suggested evidence mapping / constraint |
|---|---|
| `question` | Task instruction, without hidden tests or solution material |
| `context` / `question_context` | Permitted initial environment information and task context |
| `reasoning_trace` | Agent command/observation transcript, including stdout/stderr and exit codes; describe truncation/redaction |
| `predicted_answer` | Harness-defined attempt summary or artifact result; a command exit code alone is not benchmark success |
| `environment_feedback` | Training-time verifier result, failure explanation and permitted observations; record exactly what was exposed |
| `ground_truth` | Only a genuinely available training target. Do not fabricate a solution for terminal tasks lacking one |
| `bullets_used` | The playbook bullets supplied to reflection; keep identifiers tied to that playbook revision |

`Reflector.reflect` selects its GT template only when `use_ground_truth` is true
**and** `ground_truth` is nonempty. Otherwise it uses the no-GT template. Curator
and proposal choose solely by `use_ground_truth`; the caller must propagate a
consistent policy. Do not leak held-out verifier internals into training or
generation. Sanitizing terminal secrets is a harness responsibility, not supplied
by these synthetic tests.

## Distinguish curation phases and output semantics

- `Curator.curate`: applies operations and returns `(playbook, next_id,
  operations, call_info)`. `propose_operations`: returns `(operations, call_info)`
  without applying them. Both currently format the same GT/no-GT curator
  constants using current_step, total_samples, token_budget, playbook_stats,
  recent_reflection, current_playbook and question_context.
- `aggregate_operations`: formats the separate aggregation constant with step,
  sample count, budget, stats, current_playbook and JSON `proposed_operations`.
  Groups carry group_id, sample_numbers and operations. It has no GT toggle and
  returns operations for the caller to apply.
- Valid curator JSON requires a string `reasoning` and list `operations`; an ADD
  requires type, section and content. Proposal/reducer retain only ADD, warning
  on unsupported types. Invalid JSON or invalid operation structure returns no
  proposal/reduction. The tests verify native parser behavior, not just mocks.
- `apply_curator_operations` currently applies only ADD. Ordinary `curate` can
  report unsupported operations in its returned list even though they were not
  applied. Do not count returned UPDATE/REMOVE/DELETE entries as executed edits.
  IDs are allocated by native application, not by the model.

Thus a custom ADD/UPDATE/REMOVE prompt is not equivalent to the current native
operation contract. This is a compatibility fact, not proof that it explains
issue #41's score discrepancy. Operation extensions are separate from this guide.

## Record batch configuration

At the baseline commit, `ACEBatch._extract_config_params` defaults batch_size to
1, augmented_shuffling to true and augmented_shuffling_factor to 2. Turning
augmentation off sets the factor to 1. `curator_batch_size` remains accepted for
configuration compatibility, but proposal grouping uses `curator_num_groups` or
`max(1, floor(sqrt(original_reflection_count)))`, bounded by the augmented record
count. Groups are balanced over the augmented reflection records.

Augmentation duplicates reflection records and shuffles them; it does not create
new terminal tasks. Record the actual group assignments as well as the random
seed. Proposals read a common playbook after bullet-tag updates; nonempty groups
are sorted by group_id before reduction. With no proposed ADD operations, the
runner skips the reducer. The reducer's output is applied to the playbook.

Also capture num_epochs, max_num_rounds, playbook_token_budget, no_ground_truth,
json_mode, continue_on_llm_error, evaluation cadence, worker counts and whether
bulletpoint analysis is enabled. Do not present these implementation defaults as
the original paper's experimental settings.

## Artifacts still needed for a score reproduction

Issue #41 does not supply the official four-stage Terminal-Bench prompt text,
the exact 60/29 task IDs and ordering, pinned harness/model/container settings,
or task-level results. Obtain those artifacts before claiming matched conditions
or reproduced scores. Prompt hashes and passing offline contracts establish
traceability and interface behavior only.

Generic prompt loading is already proposed in [PR #38](https://github.com/ace-agent/ace/pull/38).
This guide and its tests introduce no competing loader, no custom paper prompts,
and no operation extension or model provider.
