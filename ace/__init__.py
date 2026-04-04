"""
ACE (Agent-Curator-Environment) System

A playbook-based learning framework with three main components:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and provides feedback
- Curator: Updates the playbook based on feedback

Usage:
    from ace import ACE
    
    ace_system = ACE(
        generator_client=client,
        reflector_client=client,
        curator_client=client,
        generator_model="model-name",
        reflector_model="model-name",
        curator_model="model-name"
    )
    
    # Offline adaptation
    results = ace_system.run(
        mode='offline',
        train_samples=train_data,
        val_samples=val_data,
        test_samples=test_data,  # Optional
        data_processor=processor,
        config=config
    )

    # Online adaptation
    results = ace_system.run(
        mode='online',
        test_samples=test_data,
        data_processor=processor,
        config=config
    )

    # Evaluation only
    results = ace_system.run(
        mode='eval_only',
        test_samples=test_data,
        data_processor=processor,
        config=config
    )
"""

from .ace import ACE
from .ace_batch import ACEBatch
from .core import Generator, Reflector, Curator, BulletpointAnalyzer

__all__ = ['ACE', 'ACEBatch', 'Generator', 'Reflector', 'Curator', 'BulletpointAnalyzer']

__version__ = "1.0.0"