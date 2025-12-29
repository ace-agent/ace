"""
Reflector agent for ACE system.
Analyzes generator outputs and provides feedback on bullet usage.
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from ..prompts.reflector import REFLECTOR_PROMPT, REFLECTOR_PROMPT_NO_GT
from llm import timed_llm_call


class Reflector:
    """
    Reflector agent that analyzes the generator's reasoning and tags
    bullets as helpful, harmful, or neutral.
    """
    
    def __init__(self, api_client, api_provider, model: str, max_tokens: int = 4096, temperature: float = 0.0):
        """
        Initialize the Reflector agent.
        
        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for reflection
            max_tokens: Maximum tokens for reflection
            temperature: Temperature for generation (0.0 to 1.0)
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        ground_truth: Optional[str],
        environment_feedback: str,
        bullets_used: str,
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "reflect",
        log_dir: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Analyze the generator's output and tag bullets.
        
        Args:
            question: The original question
            reasoning_trace: The generator's reasoning
            predicted_answer: The generator's predicted answer
            ground_truth: The ground truth answer (if available)
            environment_feedback: Feedback about correctness
            bullets_used: String representation of bullets used
            use_ground_truth: Whether to use ground truth in reflection
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging
            
        Returns:
            Tuple of (reflection_content, bullet_tags, call_info)
        """
        # Select the appropriate prompt
        if use_ground_truth and ground_truth:
            prompt = REFLECTOR_PROMPT.format(
                question,
                reasoning_trace,
                predicted_answer,
                ground_truth,
                environment_feedback,
                bullets_used
            )
        else:
            prompt = REFLECTOR_PROMPT_NO_GT.format(
                question,
                reasoning_trace,
                predicted_answer,
                environment_feedback,
                bullets_used
            )
        
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="reflector",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature
        )
        
        # Extract bullet tags
        bullet_tags = self._extract_bullet_tags(response, use_json_mode)
        
        return response, bullet_tags, call_info
    
    def _extract_bullet_tags(
        self,
        response: str,
        use_json_mode: bool
    ) -> List[Dict[str, str]]:
        """
        Extract bullet tags from reflector response.
        
        Args:
            response: The reflector's response
            use_json_mode: Whether JSON mode was used
            
        Returns:
            List of dicts with 'id' and 'tag' keys
        """
        bullet_tags = []
        
        if use_json_mode:
            try:
                response_json = json.loads(response)
                bullet_tags = response_json.get("bullet_tags", [])
            except (json.JSONDecodeError, KeyError):
                print(f"Warning: Failed to parse bullet tags from JSON response")
        else:
            # Try to extract from non-JSON response
            # This is a fallback and may not always work
            try:
                # Look for JSON-like structure in the response
                start_idx = response.find('"bullet_tags"')
                if start_idx != -1:
                    # Find the array
                    bracket_idx = response.find('[', start_idx)
                    if bracket_idx != -1:
                        # Find matching closing bracket
                        depth = 0
                        end_idx = bracket_idx
                        for i in range(bracket_idx, len(response)):
                            if response[i] == '[':
                                depth += 1
                            elif response[i] == ']':
                                depth -= 1
                                if depth == 0:
                                    end_idx = i + 1
                                    break
                        
                        bullet_tags_str = response[bracket_idx:end_idx]
                        bullet_tags = json.loads(bullet_tags_str)
            except Exception as e:
                print(f"Warning: Failed to extract bullet tags: {e}")
        
        return bullet_tags