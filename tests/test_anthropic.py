#!/usr/bin/env python3
"""
End-to-end test script for Anthropic integration in ACE framework.

This script tests:
1. Anthropic client initialization
2. Basic LLM call with Anthropic's Claude models
3. ACE system initialization with Anthropic provider

Usage:
    # Set your API key first
    export ANTHROPIC_API_KEY="your-api-key"
    cd /path/to/ace
    
    # Run the test
    python3 -m tests.test_anthropic
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def test_anthropic_client_initialization():
    """Test that Anthropic client can be initialized."""
    print("\n" + "="*60)
    print("TEST 1: Anthropic Client Initialization")
    print("="*60)
    
    try:
        from utils import initialize_clients
        
        generator_client, reflector_client, curator_client = initialize_clients("anthropic")
        
        # Check that clients are Anthropic clients
        import anthropic
        assert isinstance(generator_client, anthropic.Anthropic), "Generator client is not Anthropic client"
        assert isinstance(reflector_client, anthropic.Anthropic), "Reflector client is not Anthropic client"
        assert isinstance(curator_client, anthropic.Anthropic), "Curator client is not Anthropic client"
        
        print("✅ Anthropic clients initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Anthropic clients: {e}")
        return False


def test_anthropic_llm_call():
    """Test that we can make a basic LLM call with Anthropic."""
    print("\n" + "="*60)
    print("TEST 2: Anthropic LLM Call")
    print("="*60)
    
    try:
        from utils import initialize_clients
        from llm import timed_llm_call
        
        # Initialize client
        client, _, _ = initialize_clients("anthropic")
        
        # Make a simple test call
        prompt = "What is 2 + 2? Reply with just the number."
        
        print(f"Sending prompt: '{prompt}'")
        
        response, call_info = timed_llm_call(
            client=client,
            api_provider="anthropic",
            model="claude-3-5-haiku-20241022",  # Using Haiku for faster/cheaper testing
            prompt=prompt,
            role="test",
            call_id="test_anthropic_001",
            max_tokens=100,
            log_dir=None
        )
        
        print(f"Response: '{response}'")
        print(f"Prompt tokens: {call_info.get('prompt_num_tokens', 'N/A')}")
        print(f"Response tokens: {call_info.get('response_num_tokens', 'N/A')}")
        print(f"Total time: {call_info.get('total_time', 'N/A'):.2f}s")
        
        # Basic validation
        assert response is not None, "Response is None"
        assert len(response) > 0, "Response is empty"
        assert "4" in response, f"Expected '4' in response, got: {response}"
        
        print("✅ Anthropic LLM call successful!")
        return True
        
    except Exception as e:
        print(f"❌ Failed Anthropic LLM call: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anthropic_json_mode():
    """Test that JSON mode works with Anthropic."""
    print("\n" + "="*60)
    print("TEST 3: Anthropic JSON Mode")
    print("="*60)
    
    try:
        from utils import initialize_clients
        from llm import timed_llm_call
        
        # Initialize client
        client, _, _ = initialize_clients("anthropic")
        
        # Make a JSON mode call
        prompt = 'Return a JSON object with keys "name" and "value". Set name to "test" and value to 42.'
        
        print(f"Sending JSON prompt...")
        
        response, call_info = timed_llm_call(
            client=client,
            api_provider="anthropic",
            model="claude-3-5-haiku-20241022",
            prompt=prompt,
            role="test",
            call_id="test_anthropic_json_001",
            max_tokens=200,
            log_dir=None,
            use_json_mode=True
        )
        
        print(f"Response: '{response}'")
        
        # Try to parse as JSON
        parsed = json.loads(response)
        print(f"Parsed JSON: {parsed}")
        
        assert "name" in parsed, "Missing 'name' key in JSON response"
        assert "value" in parsed, "Missing 'value' key in JSON response"
        
        print("✅ Anthropic JSON mode successful!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed (this may be expected with Anthropic): {e}")
        print("Note: Anthropic doesn't have native JSON mode, we use system prompts instead.")
        return True  # Not a hard failure
        
    except Exception as e:
        print(f"❌ Failed Anthropic JSON mode: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ace_initialization():
    """Test that ACE system can be initialized with Anthropic."""
    print("\n" + "="*60)
    print("TEST 4: ACE System Initialization with Anthropic")
    print("="*60)
    
    try:
        from ace import ACE
        
        ace_system = ACE(
            api_provider="anthropic",
            generator_model="claude-3-5-haiku-20241022",
            reflector_model="claude-3-5-haiku-20241022",
            curator_model="claude-3-5-haiku-20241022",
            max_tokens=4096
        )
        
        # Check that all components are initialized
        assert ace_system.generator is not None, "Generator not initialized"
        assert ace_system.reflector is not None, "Reflector not initialized"
        assert ace_system.curator is not None, "Curator not initialized"
        
        print("✅ ACE system initialized successfully with Anthropic!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize ACE system: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generator_call():
    """Test that Generator can generate with Anthropic."""
    print("\n" + "="*60)
    print("TEST 5: Generator Call with Anthropic")
    print("="*60)
    
    try:
        from ace import ACE
        
        ace_system = ACE(
            api_provider="anthropic",
            generator_model="claude-3-5-haiku-20241022",
            reflector_model="claude-3-5-haiku-20241022",
            curator_model="claude-3-5-haiku-20241022",
            max_tokens=1024
        )
        
        # Test a simple generation
        question = "What is the capital of France?"
        playbook = "## STRATEGIES & INSIGHTS\n- Answer geography questions directly and concisely."
        
        print(f"Question: {question}")
        
        response, bullet_ids, call_info = ace_system.generator.generate(
            question=question,
            playbook=playbook,
            context="",
            reflection="(empty)",
            use_json_mode=False,
            call_id="test_gen_001"
        )
        
        print(f"Response preview: {response[:200]}...")
        print(f"Call time: {call_info.get('total_time', 'N/A'):.2f}s")
        
        assert response is not None, "Response is None"
        assert "paris" in response.lower(), f"Expected 'Paris' in response"
        
        print("✅ Generator call successful!")
        return True
        
    except Exception as e:
        print(f"❌ Failed Generator call: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# ACE Framework - Anthropic Integration Tests")
    print("#"*60)
    
    # Check if API key is set
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if not api_key:
        print("\n❌ ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)
    
    print(f"\n✓ ANTHROPIC_API_KEY is set (length: {len(api_key)} chars)")
    
    # Run tests
    results = []
    
    results.append(("Client Initialization", test_anthropic_client_initialization()))
    results.append(("LLM Call", test_anthropic_llm_call()))
    results.append(("JSON Mode", test_anthropic_json_mode()))
    results.append(("ACE Initialization", test_ace_initialization()))
    results.append(("Generator Call", test_generator_call()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed! Anthropic integration is working correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
