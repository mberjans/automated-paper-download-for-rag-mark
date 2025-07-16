#!/usr/bin/env python3
"""
Test Retry Behavior with Enhanced Wrapper
This script tests whether the wrapper properly implements exponential backoff
before switching providers.
"""

import time
import sys
sys.path.append('FOODB_LLM_pipeline')

def test_current_retry_behavior():
    """Test the current retry behavior"""
    print("🔄 Testing Current Retry Behavior")
    print("=" * 35)
    
    from llm_wrapper_enhanced import LLMWrapper, RetryConfig
    
    # Create wrapper with custom retry config for testing
    retry_config = RetryConfig(
        max_attempts=5,        # 5 attempts instead of 3
        base_delay=2.0,        # 2 second base delay
        max_delay=30.0,        # Max 30 seconds
        exponential_base=2.0,  # Double each time
        jitter=False           # No jitter for predictable testing
    )
    
    wrapper = LLMWrapper(retry_config=retry_config)
    
    print(f"🎯 Primary provider: {wrapper.current_provider}")
    print(f"📋 Retry config:")
    print(f"   Max attempts: {retry_config.max_attempts}")
    print(f"   Base delay: {retry_config.base_delay}s")
    print(f"   Exponential base: {retry_config.exponential_base}")
    print(f"   Max delay: {retry_config.max_delay}s")
    
    # Test with rapid requests to trigger rate limiting
    print(f"\n🚀 Sending rapid requests to trigger rate limiting...")
    
    for i in range(1, 11):
        print(f"\nRequest {i}: ", end="", flush=True)
        
        provider_before = wrapper.current_provider
        start_time = time.time()
        
        response = wrapper.generate_single(f"Test request {i}", max_tokens=50)
        
        end_time = time.time()
        provider_after = wrapper.current_provider
        
        duration = end_time - start_time
        success = len(response) > 0
        
        print(f"{'✅' if success else '❌'} {duration:.2f}s [{provider_after}]")
        
        if provider_before != provider_after:
            print(f"    🔄 Provider switch: {provider_before} → {provider_after}")
        
        if duration > 5.0:
            print(f"    ⏱️ Long delay detected: {duration:.2f}s (possible retry with backoff)")
        
        # Check statistics
        stats = wrapper.get_statistics()
        if stats['rate_limited_requests'] > 0:
            print(f"    📊 Rate limited requests: {stats['rate_limited_requests']}")
            print(f"    🔄 Fallback switches: {stats['fallback_switches']}")
    
    # Final statistics
    final_stats = wrapper.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Total requests: {final_stats['total_requests']}")
    print(f"   Successful: {final_stats['successful_requests']}")
    print(f"   Rate limited: {final_stats['rate_limited_requests']}")
    print(f"   Provider switches: {final_stats['fallback_switches']}")
    print(f"   Success rate: {final_stats['success_rate']:.1%}")
    
    return final_stats

def analyze_retry_logic():
    """Analyze the retry logic in the code"""
    print(f"\n🔍 Analyzing Current Retry Logic")
    print("=" * 35)
    
    print(f"Current Logic Flow:")
    print(f"1. Make API request")
    print(f"2. If rate limited:")
    print(f"   a. Try to switch provider immediately")
    print(f"   b. If switch successful → continue with new provider")
    print(f"   c. If no provider available → exponential backoff")
    print(f"3. If other error:")
    print(f"   a. Try to switch provider")
    print(f"   b. If no provider available → exponential backoff")
    
    print(f"\n❌ Problem:")
    print(f"   • Prioritizes provider switching over retrying")
    print(f"   • Only uses exponential backoff as last resort")
    print(f"   • Doesn't give rate-limited provider time to recover")
    
    print(f"\n✅ Better Logic Would Be:")
    print(f"1. Make API request")
    print(f"2. If rate limited:")
    print(f"   a. Try exponential backoff with same provider (2-3 attempts)")
    print(f"   b. If still failing → switch to next provider")
    print(f"3. This gives providers time to recover from rate limits")

def calculate_expected_delays():
    """Calculate what the exponential backoff delays should be"""
    print(f"\n⏱️ Expected Exponential Backoff Delays")
    print("=" * 40)
    
    from llm_wrapper_enhanced import RetryConfig
    
    # Default config
    config = RetryConfig()
    
    print(f"With default config:")
    print(f"   Base delay: {config.base_delay}s")
    print(f"   Exponential base: {config.exponential_base}")
    print(f"   Max attempts: {config.max_attempts}")
    
    print(f"\nExpected delays:")
    for attempt in range(config.max_attempts):
        delay = min(
            config.base_delay * (config.exponential_base ** attempt),
            config.max_delay
        )
        print(f"   Attempt {attempt + 1}: {delay:.1f}s delay")
    
    total_delay = sum(
        min(config.base_delay * (config.exponential_base ** attempt), config.max_delay)
        for attempt in range(config.max_attempts)
    )
    print(f"   Total delay: {total_delay:.1f}s")
    
    print(f"\n💡 In the wine PDF test:")
    print(f"   • Cerebras should have retried 3 times with delays: 1s, 2s, 4s")
    print(f"   • Total retry time: 7 seconds before switching to Groq")
    print(f"   • But it switched immediately instead")

def main():
    """Test and analyze retry behavior"""
    print("🔄 Enhanced Wrapper - Retry Behavior Analysis")
    print("=" * 50)
    
    try:
        # Test current behavior
        stats = test_current_retry_behavior()
        
        # Analyze the logic
        analyze_retry_logic()
        
        # Show expected delays
        calculate_expected_delays()
        
        print(f"\n🎯 CONCLUSION:")
        print(f"The enhanced wrapper DOES have exponential backoff logic,")
        print(f"but it prioritizes provider switching over retrying.")
        print(f"In the wine PDF test, it switched providers immediately")
        print(f"instead of trying exponential backoff with Cerebras.")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"Modify the retry logic to attempt exponential backoff")
        print(f"BEFORE switching providers for better rate limit handling.")
        
    except Exception as e:
        print(f"❌ Error testing retry behavior: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
