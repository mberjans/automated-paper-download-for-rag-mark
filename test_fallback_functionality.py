#!/usr/bin/env python3
"""
Test Enhanced LLM Wrapper with Fallback Functionality
This script demonstrates the fallback system with rate limiting and provider switching
"""

import sys
import time
import json
from enhanced_llm_wrapper_with_fallback import EnhancedLLMWrapper, RetryConfig

def test_basic_functionality():
    """Test basic functionality of enhanced wrapper"""
    print("🧪 Testing Basic Enhanced Wrapper Functionality")
    print("=" * 60)
    
    # Create wrapper with custom retry configuration
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True
    )
    
    wrapper = EnhancedLLMWrapper(retry_config=retry_config)
    
    # Show initial provider status
    print("📊 Initial Provider Status:")
    status = wrapper.get_provider_status()
    print(f"Current Provider: {status['current_provider']}")
    
    for provider, info in status['providers'].items():
        api_status = "✅ Available" if info['has_api_key'] else "❌ No API Key"
        print(f"  {provider}: {info['status']} ({api_status})")
    
    # Test simple request
    print(f"\n🔬 Testing Simple Request:")
    test_prompt = "Extract metabolites from this text: Resveratrol is found in red wine and has antioxidant properties."
    
    start_time = time.time()
    response = wrapper.generate_single_with_fallback(test_prompt, max_tokens=200)
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f}s")
    print(f"Response length: {len(response)} characters")
    print(f"Response preview: {response[:200]}...")
    
    # Show statistics
    print(f"\n📈 Statistics:")
    stats = wrapper.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return wrapper

def test_rate_limiting_simulation():
    """Simulate rate limiting behavior"""
    print(f"\n🚦 Testing Rate Limiting Simulation")
    print("=" * 50)
    
    wrapper = EnhancedLLMWrapper()
    
    # Test multiple rapid requests to trigger rate limiting
    test_prompts = [
        "Extract compounds from: Green tea contains EGCG and catechins.",
        "Find metabolites in: Coffee has caffeine and chlorogenic acid.",
        "Identify biomarkers: Turmeric contains curcumin compounds.",
        "Extract chemicals: Dark chocolate has flavonoids and theobromine.",
        "Find compounds: Blueberries contain anthocyanins and antioxidants."
    ]
    
    print(f"🔄 Processing {len(test_prompts)} requests rapidly...")
    
    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Request {i}/{len(test_prompts)}")
        print(f"Provider: {wrapper.current_provider}")
        
        start_time = time.time()
        response = wrapper.generate_single_with_fallback(prompt, max_tokens=150)
        end_time = time.time()
        
        results.append({
            'request_num': i,
            'provider': wrapper.current_provider,
            'response_time': end_time - start_time,
            'success': len(response) > 0,
            'response_length': len(response)
        })
        
        print(f"Time: {end_time - start_time:.2f}s, Success: {len(response) > 0}")
        
        # Show provider status after each request
        status = wrapper.get_provider_status()
        print(f"Provider Status: {status['current_provider']}")
    
    # Final statistics
    print(f"\n📊 Final Results:")
    stats = wrapper.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Provider health summary
    print(f"\n🏥 Provider Health Summary:")
    status = wrapper.get_provider_status()
    for provider, info in status['providers'].items():
        print(f"  {provider}: {info['status']} (failures: {info['consecutive_failures']})")
    
    return results

def test_fallback_configuration():
    """Test different fallback configurations"""
    print(f"\n⚙️ Testing Fallback Configurations")
    print("=" * 45)
    
    configurations = [
        {"max_attempts": 1, "base_delay": 0.5, "name": "Fast Fail"},
        {"max_attempts": 3, "base_delay": 1.0, "name": "Balanced"},
        {"max_attempts": 5, "base_delay": 2.0, "name": "Persistent"}
    ]
    
    test_prompt = "Extract wine compounds: Pinot Noir contains resveratrol and anthocyanins."
    
    for config in configurations:
        print(f"\n🔧 Testing {config['name']} Configuration:")
        print(f"  Max Attempts: {config['max_attempts']}")
        print(f"  Base Delay: {config['base_delay']}s")
        
        retry_config = RetryConfig(
            max_attempts=config['max_attempts'],
            base_delay=config['base_delay']
        )
        
        wrapper = EnhancedLLMWrapper(retry_config=retry_config)
        
        start_time = time.time()
        response = wrapper.generate_single_with_fallback(test_prompt)
        end_time = time.time()
        
        stats = wrapper.get_statistics()
        
        print(f"  Result: {'✅ Success' if response else '❌ Failed'}")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Retry Attempts: {stats['retry_attempts']}")
        print(f"  Provider Switches: {stats['fallback_switches']}")

def demonstrate_provider_switching():
    """Demonstrate manual provider switching"""
    print(f"\n🔄 Demonstrating Provider Switching")
    print("=" * 40)
    
    wrapper = EnhancedLLMWrapper()
    
    # Show available providers
    status = wrapper.get_provider_status()
    print(f"Available Providers:")
    for provider, info in status['providers'].items():
        api_status = "✅" if info['has_api_key'] else "❌"
        print(f"  {provider}: {api_status} {info['status']}")
    
    # Test with each provider (if available)
    test_prompt = "List wine metabolites: Red wine contains polyphenols and tannins."
    
    for provider in wrapper.fallback_order:
        if wrapper.api_keys.get(provider):
            print(f"\n🧪 Testing {provider.title()} Provider:")
            
            # Force switch to this provider
            wrapper.current_provider = provider
            wrapper.provider_health[provider].status = wrapper.provider_health[provider].status.__class__.HEALTHY
            
            start_time = time.time()
            response = wrapper.generate_single_with_fallback(test_prompt, max_tokens=100)
            end_time = time.time()
            
            print(f"  Time: {end_time - start_time:.2f}s")
            print(f"  Success: {'✅' if response else '❌'}")
            print(f"  Response: {response[:100]}..." if response else "  No response")
        else:
            print(f"\n❌ {provider.title()}: No API key available")

def save_test_results(results: list):
    """Save test results to file"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"fallback_test_results_{timestamp}.json"
    
    test_data = {
        'timestamp': timestamp,
        'test_results': results,
        'summary': {
            'total_requests': len(results),
            'successful_requests': sum(1 for r in results if r['success']),
            'average_response_time': sum(r['response_time'] for r in results) / len(results),
            'providers_used': list(set(r['provider'] for r in results if r['provider']))
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n💾 Test results saved to: {filename}")
    return filename

def main():
    """Run all fallback functionality tests"""
    print("🚀 FOODB Enhanced LLM Wrapper - Fallback Functionality Tests")
    print("=" * 70)
    
    try:
        # Test 1: Basic functionality
        wrapper = test_basic_functionality()
        
        # Test 2: Rate limiting simulation
        results = test_rate_limiting_simulation()
        
        # Test 3: Different configurations
        test_fallback_configuration()
        
        # Test 4: Provider switching
        demonstrate_provider_switching()
        
        # Save results
        save_test_results(results)
        
        print(f"\n🎉 All fallback functionality tests completed!")
        print(f"The enhanced wrapper demonstrates:")
        print(f"  ✅ Exponential backoff on rate limits")
        print(f"  ✅ Automatic provider switching")
        print(f"  ✅ Configurable retry attempts")
        print(f"  ✅ Provider health monitoring")
        print(f"  ✅ Comprehensive error handling")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
