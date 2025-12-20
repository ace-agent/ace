"""
==============================================================================
temporal_bullet.py
==============================================================================

Enhanced ACE bullet with temporal tracking for relevance-based retrieval.

This module extends ACE's playbook bullets with:
- Temporal metadata (creation time, last used, usage timeline)
- Scoring functions (recency, frequency, utility, relevance)
- Archiving logic based on staleness

"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from math import exp


@dataclass
class TemporalBullet:
    """Enhanced ACE bullet with temporal tracking"""

    # Original ACE fields
    id: str
    content: str
    bullet_type: str  # "str", "cal", "mis", etc.
    helpful_count: int = 0
    harmful_count: int = 0

    # NEW: Temporal tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    usage_timeline: List[datetime] = field(default_factory=list)

    # NEW: Context tracking
    task_types_used: List[str] = field(default_factory=list)

    # Configuration
    RECENCY_DECAY_RATE: float = 0.1  # Half-life ~7 days
    FREQUENCY_WINDOW_DAYS: int = 30

    def recency_score(self, current_time: datetime = None) -> float:
        """
        Exponential decay based on last use.
        Score ranges from 0 (very old) to 1.0 (just used).
        Half-life of ~7 days with decay_rate=0.1.
        """
        if current_time is None:
            current_time = datetime.now()

        days_since_use = (current_time - self.last_used_at).days
        return exp(-self.RECENCY_DECAY_RATE * days_since_use)

    def frequency_score(self, current_time: datetime = None) -> float:
        """
        Recent usage frequency (uses per day in last 30 days).
        Score ranges from 0 (never used) to ~1.0 (used daily).
        """
        if current_time is None:
            current_time = datetime.now()

        recent_uses = [
            t for t in self.usage_timeline
            if (current_time - t).days <= self.FREQUENCY_WINDOW_DAYS
        ]
        return len(recent_uses) / self.FREQUENCY_WINDOW_DAYS

    def utility_score(self) -> float:
        """
        Net utility based on helpful/harmful counters.
        Returns normalized score (can be negative if harmful > helpful).
        """
        return float(self.helpful_count - self.harmful_count)

    def relevance_score(self, current_time: datetime = None) -> float:
        """
        Combined relevance score for retrieval ranking.

        Formula: utility * recency * (1 + frequency)

        Rationale:
        - Base score is utility (helpful - harmful)
        - Multiplied by recency (recent knowledge weighted higher)
        - Boosted by frequency (frequently used = more relevant)
        """
        utility = self.utility_score()
        recency = self.recency_score(current_time)
        frequency = self.frequency_score(current_time)

        return utility * recency * (1 + frequency)

    def mark_used(self, task_type: str = None, current_time: datetime = None):
        """
        Record that this bullet was retrieved/used.
        Updates last_used_at, appends to usage_timeline, tracks task type.
        """
        if current_time is None:
            current_time = datetime.now()

        self.last_used_at = current_time
        self.usage_timeline.append(current_time)

        if task_type and task_type not in self.task_types_used:
            self.task_types_used.append(task_type)

    def should_archive(self,
                      min_days_inactive: int = 30,
                      current_time: datetime = None) -> bool:
        """
        Determine if bullet should be archived.

        Archive if:
        1. Not used in min_days_inactive days, OR
        2. Recency score below threshold (0.05 = ~30 days at default decay)

        Even if helpful > harmful, stale knowledge gets archived.
        """
        if current_time is None:
            current_time = datetime.now()

        days_inactive = (current_time - self.last_used_at).days
        recency = self.recency_score(current_time)

        return days_inactive >= min_days_inactive or recency < 0.05

    def to_dict(self) -> dict:
        """Serialize for storage"""
        return {
            'id': self.id,
            'content': self.content,
            'bullet_type': self.bullet_type,
            'helpful_count': self.helpful_count,
            'harmful_count': self.harmful_count,
            'created_at': self.created_at.isoformat(),
            'last_used_at': self.last_used_at.isoformat(),
            'usage_timeline': [t.isoformat() for t in self.usage_timeline],
            'task_types_used': self.task_types_used
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TemporalBullet':
        """Deserialize from storage"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used_at'] = datetime.fromisoformat(data['last_used_at'])
        data['usage_timeline'] = [datetime.fromisoformat(t) for t in data['usage_timeline']]
        return cls(**data)
