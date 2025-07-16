#!/usr/bin/env python3
"""
Diagnose Provider Selection Issue
This script checks why Groq is being selected instead of Cerebras as the primary provider
"""

import os
import sys
from enhanced_llm_wrapper_with_fallback import EnhancedLLMWrapper

def check_api_keys():
    """Check which API keys are available"""
    print("🔑 API Key Availability Check")
    print("=" * 40)
    
    api_keys = {
        'cerebras': os.getenv('CEREBRAS_API_KEY'),
        'groq': os.getenv('GROQ_API_KEY'), 
        'openrouter': os.getenv('OPENROUTER_API_KEY')
    }
    
    for provider, key in api_keys.items():
        if key:
            # Show partial key for security
            masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            print(f"  {provider}: ✅ Available ({masked_key})")
        else:
            print(f"  {provider}: ❌ Not found")
    
    return api_keys

def check_provider_initialization():
    """Check how providers are initialized"""
    print(f"\n🔧 Provider Initialization Check")
    print("=" * 40)
    
    # Create wrapper and check initialization
    wrapper = EnhancedLLMWrapper()
    
    print(f"Fallback order: {wrapper.fallback_order}")
    print(f"Selected provider: {wrapper.current_provider}")
    
    # Check provider health
    print(f"\nProvider health status:")
    for provider in wrapper.fallback_order:
        health = wrapper.provider_health.get(provider)
        has_key = bool(wrapper.api_keys.get(provider))
        print(f"  {provider}: {health.status.value if health else 'None'} (API key: {'✅' if has_key else '❌'})")
    
    return wrapper

def test_provider_selection_logic():
    """Test the provider selection logic step by step"""
    print(f"\n🧪 Provider Selection Logic Test")
    print("=" * 40)
    
    wrapper = EnhancedLLMWrapper()
    
    print("Testing _get_best_available_provider() logic:")
    
    # Manually check each provider in order
    for i, provider in enumerate(wrapper.fallback_order, 1):
        print(f"\n{i}. Checking {provider}:")
        
        health = wrapper.provider_health.get(provider)
        has_key = bool(wrapper.api_keys.get(provider))
        
        print(f"   Health status: {health.status.value if health else 'None'}")
        print(f"   Has API key: {'✅' if has_key else '❌'}")
        
        if health and health.status.value == 'healthy' and has_key:
            print(f"   ✅ This provider should be selected!")
            break
        else:
            print(f"   ❌ This provider is not available")
    
    # Test the actual method
    best_provider = wrapper._get_best_available_provider()
    print(f"\nActual _get_best_available_provider() result: {best_provider}")

def check_original_wrapper():
    """Check what the original wrapper uses"""
    print(f"\n📊 Original Wrapper Comparison")
    print("=" * 40)
    
    try:
        # Add the FOODB_LLM_pipeline directory to the path
        sys.path.append('FOODB_LLM_pipeline')
        from llm_wrapper import LLMWrapper
        
        original_wrapper = LLMWrapper()
        
        print(f"Original wrapper model: {original_wrapper.current_model.get('model_name', 'Unknown')}")
        print(f"Original wrapper provider: {original_wrapper.current_model.get('provider', 'Unknown')}")
        
        # Check if original wrapper has Cerebras configured
        if hasattr(original_wrapper, 'model_configs'):
            cerebras_configs = [m for m in original_wrapper.model_configs if 'cerebras' in m.get('provider', '').lower()]
            print(f"Cerebras configs in original: {len(cerebras_configs)}")
            
            if cerebras_configs:
                print(f"Cerebras model: {cerebras_configs[0].get('model_name', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error checking original wrapper: {e}")

def test_cerebras_connectivity():
    """Test if Cerebras is actually available"""
    print(f"\n🌐 Cerebras Connectivity Test")
    print("=" * 35)
    
    cerebras_key = os.getenv('CEREBRAS_API_KEY')
    
    if not cerebras_key:
        print("❌ No Cerebras API key found")
        return
    
    print("✅ Cerebras API key found, testing connectivity...")
    
    try:
        import requests
        
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cerebras_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 10
        }
        
        print("Making test request to Cerebras...")
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ Cerebras API is working!")
        elif response.status_code == 401:
            print("❌ Cerebras API key is invalid")
        elif response.status_code == 429:
            print("⚠️ Cerebras API is rate limited")
        else:
            print(f"❌ Cerebras API error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Error testing Cerebras: {e}")

def main():
    """Run comprehensive provider selection diagnosis"""
    print("🔍 FOODB Enhanced Wrapper - Provider Selection Diagnosis")
    print("=" * 65)
    
    # Check 1: API keys
    api_keys = check_api_keys()
    
    # Check 2: Provider initialization
    wrapper = check_provider_initialization()
    
    # Check 3: Selection logic
    test_provider_selection_logic()
    
    # Check 4: Original wrapper
    check_original_wrapper()
    
    # Check 5: Cerebras connectivity
    test_cerebras_connectivity()
    
    # Summary
    print(f"\n📋 DIAGNOSIS SUMMARY")
    print("=" * 25)
    
    cerebras_available = bool(api_keys.get('cerebras'))
    groq_available = bool(api_keys.get('groq'))
    
    if not cerebras_available and groq_available:
        print("🎯 ROOT CAUSE: Cerebras API key not available")
        print("   Solution: Set CEREBRAS_API_KEY environment variable")
        print("   Current behavior: System correctly falls back to Groq")
    elif cerebras_available and wrapper.current_provider == 'groq':
        print("🎯 ISSUE: Cerebras key available but Groq selected")
        print("   This indicates a potential bug in provider selection")
    elif cerebras_available and wrapper.current_provider == 'cerebras':
        print("✅ CORRECT: Cerebras is properly selected as primary")
    else:
        print("❓ UNCLEAR: Need to investigate further")
    
    print(f"\nCurrent provider: {wrapper.current_provider}")
    print(f"Expected provider: cerebras (if API key available)")
    print(f"Fallback working: {'✅' if wrapper.current_provider else '❌'}")

if __name__ == "__main__":
    main()
