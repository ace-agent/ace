"""
==============================================================================
temporal_curator.py
==============================================================================

Extends ACE's Curator with temporal relevance tracking.

Original ACE Curator operations:
- Semantic deduplication
- Helpful/harmful counter updates
- Pruning based on harmful > helpful

New temporal operations:
- Recency-based archiving
- Relevance-ranked retrieval
- Temporal decay analysis

"""

from typing import List, Dict
from datetime import datetime
from ace.temporal_bullet import TemporalBullet


class TemporalCurator:
    """
    Extends ACE's Curator with temporal relevance tracking.
    """

    def __init__(self,
                 playbook: List[TemporalBullet] = None,
                 archive_inactive_days: int = 30):
        self.playbook = playbook if playbook is not None else []
        self.archive = []
        self.archive_inactive_days = archive_inactive_days

    def merge_delta(self,
                    delta_bullets: List[TemporalBullet],
                    current_time: datetime = None) -> List[TemporalBullet]:
        """
        Merge delta updates into playbook (ACE's core operation).

        Enhanced with temporal tracking:
        1. Semantic deduplication (original ACE)
        2. Update helpful/harmful counters (original ACE)
        3. NEW: Update last_used_at and usage_timeline
        4. NEW: Track which task types use each bullet
        """
        if current_time is None:
            current_time = datetime.now()

        updated_playbook = []

        for delta_bullet in delta_bullets:
            # Find semantically similar bullets in existing playbook
            similar = self._find_similar(delta_bullet, self.playbook)

            if similar:
                # Update existing bullet
                similar.helpful_count += delta_bullet.helpful_count
                similar.harmful_count += delta_bullet.harmful_count
                similar.mark_used(current_time=current_time)
                updated_playbook.append(similar)
            else:
                # Add new bullet
                delta_bullet.created_at = current_time
                delta_bullet.last_used_at = current_time
                updated_playbook.append(delta_bullet)

        # Add bullets not in delta (but update their staleness)
        for bullet in self.playbook:
            if bullet not in updated_playbook:
                updated_playbook.append(bullet)

        self.playbook = updated_playbook
        return self.playbook

    def archive_stale_bullets(self, current_time: datetime = None) -> Dict:
        """
        Archive bullets that haven't been used recently.

        NEW operation not in original ACE:
        - Moves stale bullets to archive (even if helpful > harmful)
        - Preserves them for potential reactivation
        - Reduces active playbook size
        """
        if current_time is None:
            current_time = datetime.now()

        active = []
        archived = []

        for bullet in self.playbook:
            if bullet.should_archive(self.archive_inactive_days, current_time):
                self.archive.append(bullet)
                archived.append(bullet)
            else:
                active.append(bullet)

        self.playbook = active

        return {
            'archived_count': len(archived),
            'active_count': len(active),
            'archived_bullets': archived
        }

    def retrieve_relevant(self,
                         task_type: str = None,
                         top_k: int = 10,
                         current_time: datetime = None) -> List[TemporalBullet]:
        """
        Retrieve most relevant bullets based on temporal relevance.

        NEW: Relevance-based retrieval (complements original ACE's full playbook)
        - Ranks by relevance_score() = utility * recency * (1 + frequency)
        - Can filter by task_type if provided
        - Returns top_k most relevant
        """
        if current_time is None:
            current_time = datetime.now()

        # Filter by task type if specified
        candidates = self.playbook
        if task_type:
            candidates = [
                b for b in self.playbook
                if task_type in b.task_types_used or not b.task_types_used
            ]

        # Handle empty playbook
        if not candidates:
            return []

        # Rank by relevance score
        ranked = sorted(
            candidates,
            key=lambda b: b.relevance_score(current_time),
            reverse=True
        )

        # Mark retrieved bullets as used
        top_bullets = ranked[:top_k]
        for bullet in top_bullets:
            bullet.mark_used(task_type, current_time)

        return top_bullets

    def get_temporal_stats(self, current_time: datetime = None) -> Dict:
        """
        Analyze temporal patterns in playbook.

        NEW: Temporal analytics for understanding knowledge decay
        """
        if current_time is None:
            current_time = datetime.now()

        if not self.playbook:
            return {
                'total_bullets': 0,
                'avg_recency': 0,
                'avg_frequency': 0,
                'avg_relevance': 0,
                'avg_age_days': 0,
                'avg_inactive_days': 0,
                'stale_bullets': 0
            }

        recency_scores = [b.recency_score(current_time) for b in self.playbook]
        frequency_scores = [b.frequency_score(current_time) for b in self.playbook]
        relevance_scores = [b.relevance_score(current_time) for b in self.playbook]

        ages_days = [(current_time - b.created_at).days for b in self.playbook]
        inactive_days = [(current_time - b.last_used_at).days for b in self.playbook]

        return {
            'total_bullets': len(self.playbook),
            'avg_recency': sum(recency_scores) / len(recency_scores),
            'avg_frequency': sum(frequency_scores) / len(frequency_scores),
            'avg_relevance': sum(relevance_scores) / len(relevance_scores),
            'avg_age_days': sum(ages_days) / len(ages_days),
            'avg_inactive_days': sum(inactive_days) / len(inactive_days),
            'stale_bullets': sum(1 for b in self.playbook
                               if b.should_archive(self.archive_inactive_days, current_time))
        }

    def _find_similar(self,
                     bullet: TemporalBullet,
                     candidates: List[TemporalBullet],
                     threshold: float = 0.85) -> TemporalBullet:
        """
        Find semantically similar bullet (ACE's deduplication).

        Original ACE uses embedding similarity.
        Simplified here - use exact content match for now.
        TODO: Replace with embedding-based similarity for production.
        """
        # Simplified: exact content match
        for candidate in candidates:
            if candidate.content == bullet.content:
                return candidate
        return None
