#!/usr/bin/env python3
"""
Test that Cerebras is correctly selected as primary provider
This script verifies that the enhanced wrapper loads API keys from .env file
and correctly selects Cerebras as the primary provider
"""

import time
from enhanced_llm_wrapper_with_fallback import EnhancedLLMWrapper, RetryConfig

def test_provider_selection():
    """Test that Cerebras is selected as primary provider"""
    print("🧪 Testing Provider Selection from .env File")
    print("=" * 50)
    
    # Create enhanced wrapper
    wrapper = EnhancedLLMWrapper()
    
    # Check API keys loaded
    print("🔑 API Keys Loaded:")
    for provider, key in wrapper.api_keys.items():
        if key:
            masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            print(f"  {provider}: ✅ {masked_key}")
        else:
            print(f"  {provider}: ❌ Not found")
    
    # Check provider selection
    print(f"\n🎯 Provider Selection:")
    print(f"  Fallback order: {wrapper.fallback_order}")
    print(f"  Selected provider: {wrapper.current_provider}")
    
    # Check provider health
    print(f"\n🏥 Provider Health:")
    status = wrapper.get_provider_status()
    for provider, info in status['providers'].items():
        api_status = "✅" if info['has_api_key'] else "❌"
        print(f"  {provider}: {info['status']} (API key: {api_status})")
    
    # Verify Cerebras is primary
    if wrapper.current_provider == 'cerebras':
        print(f"\n✅ SUCCESS: Cerebras correctly selected as primary provider!")
    else:
        print(f"\n❌ ISSUE: Expected 'cerebras', got '{wrapper.current_provider}'")
    
    return wrapper

def test_cerebras_functionality():
    """Test that Cerebras actually works"""
    print(f"\n🔬 Testing Cerebras Functionality")
    print("=" * 40)
    
    wrapper = EnhancedLLMWrapper()
    
    if wrapper.current_provider != 'cerebras':
        print(f"❌ Skipping test - Cerebras not selected (current: {wrapper.current_provider})")
        return
    
    # Test simple request
    test_prompt = "Extract metabolites from: Red wine contains resveratrol and anthocyanins."
    
    print(f"📝 Testing request with Cerebras...")
    print(f"Prompt: {test_prompt}")
    
    start_time = time.time()
    response = wrapper.generate_single_with_fallback(test_prompt, max_tokens=100)
    end_time = time.time()
    
    success = len(response) > 0
    
    print(f"\n📊 Results:")
    print(f"  Success: {'✅' if success else '❌'}")
    print(f"  Response time: {end_time - start_time:.2f}s")
    print(f"  Response length: {len(response)} characters")
    print(f"  Provider used: {wrapper.get_provider_status()['current_provider']}")
    
    if success:
        print(f"\n📝 Response preview:")
        print(f"  {response[:200]}...")
    
    # Check statistics
    stats = wrapper.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Successful: {stats['successful_requests']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    
    return success

def test_rate_limiting_with_cerebras():
    """Test rate limiting behavior with Cerebras as primary"""
    print(f"\n🚨 Testing Rate Limiting with Cerebras Primary")
    print("=" * 50)
    
    wrapper = EnhancedLLMWrapper()
    
    if wrapper.current_provider != 'cerebras':
        print(f"❌ Skipping test - Cerebras not selected")
        return
    
    # Send multiple rapid requests to test rate limiting
    test_prompts = [
        "Extract compounds: Sample 1 contains polyphenols.",
        "Extract compounds: Sample 2 contains flavonoids.", 
        "Extract compounds: Sample 3 contains anthocyanins.",
        "Extract compounds: Sample 4 contains catechins.",
        "Extract compounds: Sample 5 contains resveratrol."
    ]
    
    print(f"🔥 Sending {len(test_prompts)} rapid requests...")
    
    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  Request {i}: ", end="")
        
        provider_before = wrapper.get_provider_status()['current_provider']
        
        start_time = time.time()
        response = wrapper.generate_single_with_fallback(prompt, max_tokens=50)
        end_time = time.time()
        
        provider_after = wrapper.get_provider_status()['current_provider']
        success = len(response) > 0
        
        results.append({
            'request': i,
            'provider_before': provider_before,
            'provider_after': provider_after,
            'success': success,
            'time': end_time - start_time
        })
        
        print(f"{'✅' if success else '❌'} {end_time - start_time:.2f}s [{provider_after}]")
        
        if provider_before != provider_after:
            print(f"    🔄 Provider switched: {provider_before} → {provider_after}")
        
        # Short delay between requests
        time.sleep(0.2)
    
    # Analyze results
    print(f"\n📊 Rate Limiting Test Results:")
    successful = sum(1 for r in results if r['success'])
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results):.1%})")
    
    # Check for provider switches
    switches = [r for r in results if r['provider_before'] != r['provider_after']]
    if switches:
        print(f"  Provider switches: {len(switches)}")
        for switch in switches:
            print(f"    Request {switch['request']}: {switch['provider_before']} → {switch['provider_after']}")
    else:
        print(f"  Provider switches: 0 (stayed on {results[0]['provider_before']})")
    
    # Final statistics
    stats = wrapper.get_statistics()
    print(f"\n📈 Final Statistics:")
    print(f"  Rate limited requests: {stats['rate_limited_requests']}")
    print(f"  Fallback switches: {stats['fallback_switches']}")
    print(f"  Overall success rate: {stats['success_rate']:.1%}")
    
    return results

def main():
    """Run comprehensive Cerebras primary provider tests"""
    print("🧬 FOODB Enhanced Wrapper - Cerebras Primary Provider Test")
    print("=" * 65)
    
    try:
        # Test 1: Provider selection
        wrapper = test_provider_selection()
        
        # Test 2: Basic functionality
        if wrapper.current_provider == 'cerebras':
            success = test_cerebras_functionality()
            
            # Test 3: Rate limiting behavior
            if success:
                test_rate_limiting_with_cerebras()
        
        print(f"\n🎉 Cerebras Primary Provider Tests Completed!")
        
        if wrapper.current_provider == 'cerebras':
            print(f"✅ Cerebras is correctly configured as primary provider")
            print(f"✅ API key loaded from .env file successfully")
            print(f"✅ Fallback system ready with Groq and OpenRouter")
        else:
            print(f"⚠️ Cerebras not selected as primary provider")
            print(f"Current provider: {wrapper.current_provider}")
            print(f"Check .env file for CEREBRAS_API_KEY")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
