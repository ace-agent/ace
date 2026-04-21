"""
PromptConfig dataclass for ACE system.
Holds all agent prompts in a single configuration object.
"""

from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration dataclass holding all agent prompts."""
    generator_prompt: str
    reflector_prompt: str
    reflector_prompt_no_gt: str
    curator_prompt: str
    curator_prompt_no_gt: str
