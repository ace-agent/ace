"""Opt-in playbook transactions with stable targets and immutable candidates.

These pure helpers do not mutate an ACE instance. The synchronized training
boundary publishes both returned values together. They are not a cross-process
lock or a persistent transaction store.
"""
from dataclasses import dataclass
import re

from playbook_utils import format_playbook_line, get_next_global_id, parse_playbook_line
from utils import count_tokens, get_section_slug


@dataclass(frozen=True)
class PlaybookCandidate:
    base_playbook: str
    base_next_id: int
    playbook: str
    next_id: int
    operation_count: int


def _section_key(name):
    key = '_'.join(name.strip().lower().replace('&', 'and').split())
    return re.sub(r'_+', '_', key).strip('_')


def _content(value):
    if not isinstance(value, str) or not value.strip() or '\n' in value or '\r' in value:
        raise ValueError('content must be a non-empty single-line string')
    return value


def prepare_playbook_candidate(playbook_text, operations, next_id):
    """Validate the whole batch against its base, then construct a frozen result.

    Targets are base bullet IDs, never offsets or IDs allocated in this batch.
    Each base target may occur once. MERGE keeps the first source's ID/position
    and sums historical counters; UPDATE preserves its target's counters.
    """
    if not isinstance(playbook_text, str) or not isinstance(operations, list):
        raise ValueError('playbook must be text and operations must be a list')
    if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 1:
        raise ValueError('next_id must be a positive integer')
    if next_id < get_next_global_id(playbook_text):
        raise ValueError('next_id collides with an existing global ID')

    lines = playbook_text.split('\n')
    section = 'general'
    sections = {}
    bullets = {}
    for index, line in enumerate(lines):
        if line.strip().startswith('##'):
            section = _section_key(line.strip()[2:])
            if not section or section in sections:
                raise ValueError('section headers must be non-empty and unique')
            sections[section] = index + 1
        parsed = parse_playbook_line(line)
        if parsed:
            if not sections:
                sections['general'] = 0
            if parsed['id'] in bullets:
                raise ValueError('base bullet IDs must be unique')
            bullets[parsed['id']] = (index, section, parsed)

    if not sections:
        sections['general'] = 0

    touched = set()
    replacements = {}
    deleted = set()
    additions = []
    allocated_next_id = next_id
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError('each operation must be a dictionary')
        kind = operation.get('type')
        if kind not in ('ADD', 'UPDATE', 'MERGE', 'DELETE'):
            raise ValueError(f'unsupported operation: {kind}')
        if kind == 'ADD':
            raw_section = operation.get('section')
            if not isinstance(raw_section, str) or _section_key(raw_section) not in sections:
                raise ValueError('ADD must name an existing section')
            section = _section_key(raw_section)
            content = _content(operation.get('content'))
            bullet_id = f'{get_section_slug(section)}-{allocated_next_id:05d}'
            if bullet_id in bullets:
                raise ValueError('allocated bullet ID collides with base')
            additions.append((section, format_playbook_line(bullet_id, 0, 0, content)))
            allocated_next_id += 1
            continue

        targets = operation.get('source_ids') if kind == 'MERGE' else [operation.get('bullet_id')]
        if not isinstance(targets, list) or (kind == 'MERGE' and len(targets) < 2):
            raise ValueError('MERGE requires at least two source_ids')
        if any(not isinstance(target, str) or target not in bullets for target in targets):
            raise ValueError('all targets must identify existing base bullets')
        if len(set(targets)) != len(targets) or touched.intersection(targets):
            raise ValueError('each base target may be touched only once per batch')
        touched.update(targets)
        records = [bullets[target] for target in targets]
        if kind == 'DELETE':
            deleted.add(records[0][0])
            continue
        content = _content(operation.get('content'))
        if kind == 'MERGE' and len({record[1] for record in records}) != 1:
            raise ValueError('MERGE sources must belong to the same section')
        first_index, _, first = records[0]
        replacements[first_index] = format_playbook_line(
            first['id'], sum(record[2]['helpful'] for record in records),
            sum(record[2]['harmful'] for record in records), content,
        )
        deleted.update(record[0] for record in records[1:])

    # ADD insertion positions are computed from the base, so target order cannot
    # make an earlier addition available to a later modification.
    insertions = {}
    for section, line in additions:
        start = sections[section]
        end = next((i for i in range(start, len(lines)) if lines[i].strip().startswith('##')), len(lines))
        if end == len(lines) and lines[-1] == '':
            end -= 1
        insertions.setdefault(end, []).append(line)
    result = []
    for index, line in enumerate(lines):
        result.extend(insertions.get(index, []))
        if index not in deleted:
            result.append(replacements.get(index, line))
    result.extend(insertions.get(len(lines), []))
    return PlaybookCandidate(playbook_text, next_id, '\n'.join(result), allocated_next_id, len(operations))


def publish_playbook_candidate(candidate, current_playbook, current_next_id, token_budget, enforce=False):
    """Return the candidate only if its base and optional final budget still match."""
    decision = {'status': 'rejected', 'operation_count': candidate.operation_count,
                'enforced': bool(enforce), 'reason': 'stale_base'}
    if (current_playbook, current_next_id) != (candidate.base_playbook, candidate.base_next_id):
        return current_playbook, current_next_id, decision
    if enforce and (isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 0):
        decision['reason'] = 'invalid_budget'
        return current_playbook, current_next_id, decision
    base_tokens = count_tokens(current_playbook)
    candidate_tokens = count_tokens(candidate.playbook)
    decision.update(base_tokens=base_tokens, candidate_tokens=candidate_tokens,
                    budget_tokens=token_budget if enforce else None)
    if enforce and candidate_tokens > token_budget and candidate.operation_count:
        decision['reason'] = 'candidate_exceeds_budget'
        return current_playbook, current_next_id, decision
    decision.update(status='accepted', reason='no_operations' if not candidate.operation_count else 'validated_candidate')
    return candidate.playbook, candidate.next_id, decision


def rollback_playbook_candidate(candidate, current_playbook, current_next_id):
    """Restore a published candidate's base without overwriting subsequent work."""
    if (current_playbook, current_next_id) != (candidate.playbook, candidate.next_id):
        return current_playbook, current_next_id, {'status': 'rejected', 'reason': 'stale_candidate'}
    return candidate.base_playbook, candidate.base_next_id, {'status': 'rolled_back', 'reason': 'candidate_rollback'}


def apply_playbook_lifecycle(playbook_text, operations, next_id, token_budget, enforce=False):
    """Convenience boundary: invalid input leaves text and ID untouched."""
    try:
        candidate = prepare_playbook_candidate(playbook_text, operations, next_id)
    except ValueError as error:
        return playbook_text, next_id, {'status': 'rejected', 'reason': 'invalid_operations', 'detail': str(error)}
    return publish_playbook_candidate(candidate, playbook_text, next_id, token_budget, enforce)
