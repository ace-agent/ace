"""
Curator prompts for ACE system.
"""

# Curator prompt for intelligent playbook management
CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth. 

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}

---
"""

CURATOR_PROMPT_NO_GT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}

---
"""


CURATOR_OPERATIONS_AGGREGATION_PROMPT = """You are aggregating playbook update proposals generated independently from different subsets of a training batch.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Context:**
- Each proposal below was produced by a Curator after reviewing a different reflection group.
- The current ACE implementation only supports ADD operations. Do not output UPDATE, MERGE, DELETE, or CREATE_META operations.
- Bullet IDs are assigned by the system. Do not include bullet IDs in operation content.

**Instructions:**
- Review the current playbook and all proposed ADD operations.
- Synthesize overlapping proposals into a single more specific ADD operation.
- Drop proposals that are already covered by the current playbook.
- Drop vague, redundant, or low-value proposals.
- Preserve complementary insights as separate ADD operations when they teach genuinely different behavior.
- Keep each operation concise, actionable, and suitable for future tasks.
- If no new content should be added, return an empty operations list.

**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Current Playbook:**
{current_playbook}

**Independent Proposed Operations:**
{proposed_operations}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: a concise rationale for the aggregation decisions
- operations: a list of final ADD operations to apply to the playbook
  - type: must be "ADD"
  - section: the section to add the bullet to
  - content: the final bullet content; do not include a bullet ID

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Concise rationale here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "formulas_and_calculations",
      "content": "[Deduplicated, actionable insight...]"
    }}
  ]
}}

---
"""


LIFECYCLE_OPERATIONS_SCHEMA = """Available operations (use stable IDs from the current playbook):
- ADD: {"type": "ADD", "section": "an existing section", "content": "new advice"}
- UPDATE: {"type": "UPDATE", "bullet_id": "existing-id", "content": "corrected advice"}
- DELETE: {"type": "DELETE", "bullet_id": "existing-id"}
- MERGE: {"type": "MERGE", "source_ids": ["first-id", "second-id"], "content": "combined advice"}
Every content value must be a non-empty single line, without a bullet header.
Each target must exist in the base playbook and may be touched only once in the
final batch. MERGE needs at least two distinct sources from the same section;
it keeps the first source ID and position, sums their historical counters and
removes the other sources. UPDATE preserves its ID and historical counters.
ADD allocates a fresh ID; DELETE and MERGE never reuse or allocate IDs.
Do not target an ADD from this batch. Do not output CREATE_META or other types.
Return ONLY JSON with a string reasoning and a list operations. Use [] when no
change is justified. All operations are validated and published as one batch;
a single invalid target, conflicting operation or exceeded enforced budget
rejects the entire batch. Do not delete bullets solely because they are unused;
provide a concise evidence-based rationale for changes.
"""

CURATOR_LIFECYCLE_PROMPT = """Curate the existing playbook using evidence from the recent reflection.
The reflection is generated using {reflection_source}, unavailable to future
playbook users. Correct inaccurate advice, consolidate overlapping advice,
remove demonstrated harmful advice and add useful missing advice. Keep changes
concise and actionable. Do not regenerate the entire playbook.

Training progress: Sample {current_step} out of {total_samples}
Total playbook token budget: {token_budget}
Current playbook stats:
{playbook_stats}
Recent reflection:
{recent_reflection}
Current playbook:
{current_playbook}
Question context:
{question_context}

{operation_schema}
"""

CURATOR_LIFECYCLE_AGGREGATION_PROMPT = """Reduce independent playbook change proposals into one coherent batch.
All groups used the same current playbook. Reconcile duplicate or conflicting
targets before returning final operations. Keep useful complementary changes,
remove unsupported or redundant proposals, and make conservative changes
supported by the proposal evidence. Do not apply proposals independently.

Training progress: Sample {current_step} out of {total_samples}
Total playbook token budget: {token_budget}
Current playbook stats:
{playbook_stats}
Current playbook:
{current_playbook}
Independent proposed operations:
{proposed_operations}

{operation_schema}
"""
