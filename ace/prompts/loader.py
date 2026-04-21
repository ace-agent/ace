"""
Prompt loader for ACE system.
Provides functionality to load task-specific prompts with fallback to defaults.
"""

import os
import importlib.util
from typing import Optional

from .config import PromptConfig
from .generator import GENERATOR_PROMPT
from .reflector import REFLECTOR_PROMPT, REFLECTOR_PROMPT_NO_GT
from .curator import CURATOR_PROMPT, CURATOR_PROMPT_NO_GT


def _load_prompt_from_file(file_path: str, variable_name: str) -> Optional[str]:
    """
    Load a prompt variable from a Python file.

    Args:
        file_path: Path to the Python file containing the prompt
        variable_name: Name of the variable to load (e.g., 'GENERATOR_PROMPT')

    Returns:
        The prompt string if found, None otherwise
    """
    if not os.path.exists(file_path):
        return None

    try:
        spec = importlib.util.spec_from_file_location("prompt_module", file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, variable_name):
            return getattr(module, variable_name)
        return None
    except Exception as e:
        print(f"Warning: Failed to load {variable_name} from {file_path}: {e}")
        return None


def load_prompts(task_prompts_dir: Optional[str] = None) -> PromptConfig:
    """
    Load prompts with task-specific overrides.

    Checks if task-specific prompts exist in the given directory and loads them.
    Falls back to default prompts for any missing prompts.

    Args:
        task_prompts_dir: Path to task's prompts directory (e.g., "eval/my_task/prompts")
                         If None or doesn't exist, uses all defaults.

    Returns:
        PromptConfig with loaded prompts (custom where available, defaults otherwise)

    Example:
        # Load with task-specific prompts
        config = load_prompts("eval/finance/prompts")

        # Load with all defaults
        config = load_prompts()
    """
    # Start with default prompts
    generator_prompt = GENERATOR_PROMPT
    reflector_prompt = REFLECTOR_PROMPT
    reflector_prompt_no_gt = REFLECTOR_PROMPT_NO_GT
    curator_prompt = CURATOR_PROMPT
    curator_prompt_no_gt = CURATOR_PROMPT_NO_GT

    # If task_prompts_dir is provided and exists, try to load custom prompts
    if task_prompts_dir and os.path.isdir(task_prompts_dir):
        # Try to load generator prompt
        generator_file = os.path.join(task_prompts_dir, "generator.py")
        custom_generator = _load_prompt_from_file(generator_file, "GENERATOR_PROMPT")
        if custom_generator is not None:
            generator_prompt = custom_generator
            print(f"Loaded custom GENERATOR_PROMPT from {generator_file}")

        # Try to load reflector prompts
        reflector_file = os.path.join(task_prompts_dir, "reflector.py")
        custom_reflector = _load_prompt_from_file(reflector_file, "REFLECTOR_PROMPT")
        if custom_reflector is not None:
            reflector_prompt = custom_reflector
            print(f"Loaded custom REFLECTOR_PROMPT from {reflector_file}")

        custom_reflector_no_gt = _load_prompt_from_file(reflector_file, "REFLECTOR_PROMPT_NO_GT")
        if custom_reflector_no_gt is not None:
            reflector_prompt_no_gt = custom_reflector_no_gt
            print(f"Loaded custom REFLECTOR_PROMPT_NO_GT from {reflector_file}")

        # Try to load curator prompts
        curator_file = os.path.join(task_prompts_dir, "curator.py")
        custom_curator = _load_prompt_from_file(curator_file, "CURATOR_PROMPT")
        if custom_curator is not None:
            curator_prompt = custom_curator
            print(f"Loaded custom CURATOR_PROMPT from {curator_file}")

        custom_curator_no_gt = _load_prompt_from_file(curator_file, "CURATOR_PROMPT_NO_GT")
        if custom_curator_no_gt is not None:
            curator_prompt_no_gt = custom_curator_no_gt
            print(f"Loaded custom CURATOR_PROMPT_NO_GT from {curator_file}")

    return PromptConfig(
        generator_prompt=generator_prompt,
        reflector_prompt=reflector_prompt,
        reflector_prompt_no_gt=reflector_prompt_no_gt,
        curator_prompt=curator_prompt,
        curator_prompt_no_gt=curator_prompt_no_gt
    )
