"""
Prompts module for ACE system.
Contains all prompts for Generator, Reflector, and Curator agents.
"""

from .generator import *
from .reflector import *
from .curator import *
from .config import PromptConfig
from .loader import load_prompts

__all__ = [
    # Generator prompts
    'GENERATOR_PROMPT',

    # Reflector prompts
    'REFLECTOR_PROMPT',
    'REFLECTOR_PROMPT_NO_GT',

    # Curator prompts
    'CURATOR_PROMPT',
    'CURATOR_PROMPT_NO_GT',

    # Prompt configuration
    'PromptConfig',
    'load_prompts',
]