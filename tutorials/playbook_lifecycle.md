# Optional playbook lifecycle transactions

This Draft extends the optional budget guardrail from PR #44. The lifecycle API
and merge semantics are proposed interfaces for issue #26, pending maintainer
review. It does not claim improved benchmark accuracy or training stability.

Enable it in the ACE or ACEBatch training configuration:

```python
config = {
    'enable_playbook_lifecycle': True,
    'enforce_playbook_token_budget': True,
    'playbook_token_budget': 80000,
    'use_bulletpoint_analyzer': False,
}
```

The finance, mind2web and mind2web2 entry points also accept
`--enable_playbook_lifecycle`. Both new options are false by default. Leaving
lifecycle disabled retains the native ADD-only prompts and operation behavior.
The legacy BulletpointAnalyzer cannot be combined with lifecycle mode: its
independent mutation would bypass target validation and the final budget gate,
so configuration rejects that combination explicitly.

## Operation contract

All targets refer to stable IDs in the playbook supplied to the Curator. The
entire operation batch is validated before it can change playbook text or the
next ID. Supported schemas are:

```json
[
  {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Record the exit code."},
  {"type": "UPDATE", "bullet_id": "str-00001", "content": "Inspect stderr before retrying."},
  {"type": "DELETE", "bullet_id": "str-00003"},
  {"type": "MERGE", "source_ids": ["str-00005", "str-00006"], "content": "Read both error sources before retrying."}
]
```

- ADD requires an existing section, uses the existing section slug, and allocates
  a fresh global ID. Unknown sections are rejected in this opt-in mode.
- UPDATE preserves the target ID, position and historical helpful/harmful
  counters. Those counts describe its lineage; they are not evidence that the
  replacement wording has been evaluated.
- DELETE removes only the target and does not advance or decrease the next ID.
- MERGE requires at least two distinct IDs from one section. It retains the
  **first source ID and position**, sums historical helpful/harmful counts and
  removes the remaining sources. It does not allocate an ID. Summed counters
  describe source histories, not independent evaluation of the merged wording.

Each target may be touched once per batch, including all MERGE sources. UPDATE
plus DELETE of the same ID, overlapping MERGEs, unknown IDs, duplicate base IDs,
ambiguous section headers, invalid next IDs, unsupported types, empty/multiline
content and cross-section MERGEs reject the whole batch. A newly added bullet
cannot be targeted by another operation in the same batch. There is no automatic
pruning rule or `CREATE_META` operation. The Curator must justify changes from
reflection evidence; unused bullets are not removed merely for being unused.

## Preparation, publication and rollback

`playbook_lifecycle.py` exposes pure helpers:

```python
from playbook_lifecycle import (
    prepare_playbook_candidate,
    publish_playbook_candidate,
    rollback_playbook_candidate,
)

candidate = prepare_playbook_candidate(playbook, operations, next_id)
# candidate is frozen and retains only text/scalars, not operation dictionaries.
playbook, next_id, decision = publish_playbook_candidate(
    candidate, playbook, next_id, token_budget=80000, enforce=True,
)
# On a downstream failure, guarded rollback restores both values:
playbook, next_id, rollback = rollback_playbook_candidate(candidate, playbook, next_id)
```

Publication checks both the base text and next ID, rejects stale candidates,
and applies the optional budget to final playbook text. An over-budget or
invalid batch returns the exact original text/ID without truncation or partial
application. An empty valid batch preserves bytes without a new allocation;
like the budget guardrail, an unchanged already-over-budget playbook is a no-op,
while an invalid budget is still rejected. A non-empty batch must finish within
the enforced budget, including shrinking operations on a bloated playbook.

Rollback restores the candidate's base only if current text and ID still match
that candidate's published state. It refuses to overwrite later changes.
These functions provide an in-memory contract at ACE's synchronized curation
boundary; they do not supply a lock for external concurrent writers or a durable
transaction database.

Ordinary ACE restores its pre-curation text/ID if post-curation generation
raises. ACEBatch restores its entire pre-Phase2 snapshot (including Phase2
counter updates) for a post-curation generation exception, whether training
continues after that failure or propagates it. Batch proposals use one shared
base; the reducer reconciles conflicts and publishes exactly one final batch.
Quality regressions and successful model responses do not automatically trigger
rollback. Existing best-playbook selection remains separate.

Curator `call_info` and training tracking expose `playbook_lifecycle`, including
accepted/rejected status and reason; budget decisions include base/candidate/
budget token counts. Continued batch failures also include a rollback decision.

## Offline validation

The small import environment requires `openai`, `tiktoken`, `python-dotenv` and
`numpy`; optional embedding packages are not exercised. Install dependencies and
cache the `cl100k_base` tokenizer before running without network access.

```bash
python -m unittest discover -s tests -v
python -m compileall -q ace playbook_lifecycle.py eval tests
```

Tests use synthetic playbooks and deterministic replacements at the model-call
boundary. They cover stable identities/counters, invalid/overlapping targets,
immutability, stale publication/rollback, exact budget boundary, rejection,
actual ACE/ACEBatch publication and post-generation exception rollback. No API
client, real model, embedding download or production dataset is needed. Live
provider and benchmark-quality evaluation remain future integration work.
