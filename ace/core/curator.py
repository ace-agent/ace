"""
Curator agent for ACE system.
Manages playbook operations (ADD, UPDATE, MERGE, DELETE).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from ..prompts.curator import CURATOR_PROMPT, CURATOR_PROMPT_NO_GT
from playbook_utils import extract_json_from_text, apply_curator_operations
from logger import log_curator_operation_diff, log_curator_failure
from llm import timed_llm_call

class Curator:
    """
    Curator agent that manages the playbook by adding, updating,
    merging, and deleting bullets based on reflection feedback.
    """
    
    def __init__(self, api_client, api_provider, model: str, max_tokens: int = 4096):
        """
        Initialize the Curator agent.
        
        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for curation
            max_tokens: Maximum tokens for curation
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
    
    def curate(
        self,
        current_playbook: str,
        recent_reflection: str,
        question_context: str,
        current_step: int,
        total_samples: int,
        token_budget: int,
        playbook_stats: Dict[str, Any],
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "curate",
        log_dir: Optional[str] = None,
        next_global_id: int = 1
    ) -> Tuple[str, int, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Curate the playbook based on reflection feedback.
        
        Args:
            current_playbook: Current playbook content
            recent_reflection: Recent reflection from reflector
            question_context: Context for the current question
            current_step: Current training step
            total_samples: Total number of training samples
            token_budget: Total token budget for playbook
            playbook_stats: Statistics about current playbook
            use_ground_truth: Whether ground truth is available
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging
            next_global_id: Next available global ID for bullets
            
        Returns:
            Tuple of (updated_playbook, next_global_id, operations, call_info)
        """
        # Format playbook stats as JSON string
        stats_str = json.dumps(playbook_stats, indent=2)
        
        # Select the appropriate prompt
        if use_ground_truth:
            prompt = CURATOR_PROMPT.format(
                current_step=current_step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats_str,
                recent_reflection=recent_reflection,
                current_playbook=current_playbook,
                question_context=question_context
            )
        else:
            prompt = CURATOR_PROMPT_NO_GT.format(
                current_step=current_step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats_str,
                recent_reflection=recent_reflection,
                current_playbook=current_playbook,
                question_context=question_context
            )
        
        # Make the LLM call
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="curator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode
        )
        
        # Check for empty response error
        if response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
            print(f"⏭️  Skipping curator operation due to empty response")
            log_curator_failure(log_dir, current_step, "empty_response", 
                                    response[:200], 0)
            return current_playbook, next_global_id, [], call_info
        
        # Extract and validate operations
        try:
            operations_info = self._extract_and_validate_operations(response)
            
            operations = operations_info["operations"]
            print(f"✅ Curator JSON schema validated successfully: {len(operations)} operations")
            
            # Log detailed diff for each operation before applying
            for op in operations:
                try:
                    log_curator_operation_diff(Path(log_dir).parent, op, current_playbook, call_id)
                except Exception as e:
                    print(f"Warning: Failed to log curator operation diff: {e}")
            
            # Apply operations to playbook
            updated_playbook, next_global_id = apply_curator_operations(
                current_playbook, operations, next_global_id
            )
            
            # Log operations
            for op in operations:
                try:
                    op_type = op.get('type', 'UNKNOWN') if isinstance(op, dict) else 'INVALID'
                    op_reason = op.get('content', 'No reason given') if isinstance(op, dict) else 'Invalid operation format'
                    print(f"  - {op_type}: {op_reason}")
                except Exception as e:
                    print(f"  - UNKNOWN: Error logging operation: {e}")
            
            return updated_playbook, next_global_id, operations, call_info
            
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"❌ Curator JSON parsing failed: {e}")
            print(f"📄 Raw curator response preview: {response[:300]}...")
            
            log_curator_failure(log_dir, current_step, "json_parse_error", 
                                response, 0, str(e))
            
            print("⏭️  Skipping curator operation due to invalid JSON format")
            return current_playbook, next_global_id, [], call_info
            
        except Exception as e:
            print(f"❌ Curator operation failed: {e}")
            print(f"📄 Raw curator response preview: {response[:300]}...")
            
            log_curator_failure(log_dir, current_step, "operation_error", 
                                response, 0, str(e))
            
            print("⏭️  Skipping curator operation and continuing training")
            return current_playbook, next_global_id, [], call_info
    
    def _extract_and_validate_operations(
        self,
        response: str
    ) -> Dict[str, Any]:
        """
        Extract and validate operations from curator response.
        
        Args:
            response: The curator's response
            
        Returns:
            Dictionary with 'reasoning' and 'operations' keys
            
        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        # Extract operations info
        operations_info = extract_json_from_text(response, "operations")
        
        # Validate JSON structure is correct
        if not operations_info:
            raise ValueError("Failed to extract valid JSON from curator response")
        
        # Validate required fields
        if "reasoning" not in operations_info:
            raise ValueError("JSON missing required 'reasoning' field")
        
        if "operations" not in operations_info:
            raise ValueError("JSON missing required 'operations' field")
        
        # Validate field types
        if not isinstance(operations_info["reasoning"], str):
            raise ValueError("'reasoning' field must be a string")
        
        if not isinstance(operations_info["operations"], list):
            raise ValueError("'operations' field must be a list")
        
        # Validate operations structure
        for i, op in enumerate(operations_info["operations"]):
            if not isinstance(op, dict):
                raise ValueError(f"Operation {i} must be a dictionary")
            
            if "type" not in op:
                raise ValueError(f"Operation {i} missing required 'type' field")
            
            op_type = op["type"]
            
            # Currently only ADD operations are fully supported
            # Note: You can add support for UPDATE, MERGE, DELETE operations here
            if op_type not in ["ADD", "UPDATE", "MERGE", "DELETE", "CREATE_META"]:
                print(f"Warning: Operation type '{op_type}' may not be fully supported")
            
            # Validate ADD operation structure
            if op_type == "ADD":
                required_fields = {"type", "section", "content"}
                missing_fields = required_fields - set(op.keys())
                if missing_fields:
                    raise ValueError(f"ADD operation {i} missing fields: {list(missing_fields)}")
        
        return operations_info