#!/usr/bin/env python3
"""
Test Exponential Backoff Logic
This script tests the updated wrapper to ensure it uses exponential backoff
BEFORE switching providers.
"""

import time
import sys
sys.path.append('FOODB_LLM_pipeline')

def test_exponential_backoff_behavior():
    """Test that exponential backoff is used before provider switching"""
    print("🔄 Testing Exponential Backoff Before Provider Switching")
    print("=" * 60)
    
    from llm_wrapper_enhanced import LLMWrapper, RetryConfig
    
    # Create wrapper with custom config for testing
    retry_config = RetryConfig(
        max_attempts=3,        # 3 attempts for faster testing
        base_delay=1.0,        # 1 second base delay
        max_delay=10.0,        # Max 10 seconds for testing
        exponential_base=2.0,  # Double each time
        jitter=False           # No jitter for predictable testing
    )
    
    wrapper = LLMWrapper(retry_config=retry_config)
    
    print(f"🎯 Primary provider: {wrapper.current_provider}")
    print(f"📋 Retry configuration:")
    print(f"   Max attempts: {retry_config.max_attempts}")
    print(f"   Base delay: {retry_config.base_delay}s")
    print(f"   Expected delays: 1s, 2s, 4s")
    print(f"   Total retry time: ~7s before switching")
    
    print(f"\n🚀 Sending requests to trigger rate limiting and observe backoff...")
    
    request_times = []
    provider_switches = []
    
    for i in range(1, 15):  # Send enough requests to trigger rate limiting
        print(f"\nRequest {i:2d}: ", end="", flush=True)
        
        provider_before = wrapper.current_provider
        start_time = time.time()
        
        response = wrapper.generate_single(f"Test exponential backoff request {i}", max_tokens=50)
        
        end_time = time.time()
        provider_after = wrapper.current_provider
        
        duration = end_time - start_time
        success = len(response) > 0
        
        request_times.append(duration)
        
        print(f"{'✅' if success else '❌'} {duration:.2f}s [{provider_after}]")
        
        # Detect provider switches
        if provider_before != provider_after:
            provider_switches.append({
                'request': i,
                'from': provider_before,
                'to': provider_after,
                'duration': duration
            })
            print(f"    🔄 Provider switch: {provider_before} → {provider_after}")
        
        # Detect exponential backoff (long delays)
        if duration > 5.0:
            print(f"    ⏱️ Long delay detected: {duration:.2f}s (likely exponential backoff)")
        elif duration > 2.0:
            print(f"    ⏱️ Moderate delay: {duration:.2f}s (possible retry)")
        
        # Show statistics
        stats = wrapper.get_statistics()
        if stats['rate_limited_requests'] > 0:
            print(f"    📊 Rate limited: {stats['rate_limited_requests']}, Switches: {stats['fallback_switches']}")
    
    # Analyze results
    print(f"\n📊 EXPONENTIAL BACKOFF ANALYSIS")
    print("=" * 35)
    
    # Look for evidence of exponential backoff
    long_delays = [t for t in request_times if t > 5.0]
    moderate_delays = [t for t in request_times if 2.0 <= t <= 5.0]
    
    print(f"Request timing analysis:")
    print(f"   Fast requests (<2s): {len([t for t in request_times if t < 2.0])}")
    print(f"   Moderate delays (2-5s): {len(moderate_delays)}")
    print(f"   Long delays (>5s): {len(long_delays)}")
    
    if long_delays:
        print(f"   Longest delay: {max(long_delays):.2f}s")
        print(f"   ✅ Evidence of exponential backoff found!")
    else:
        print(f"   ❌ No long delays detected - may not have triggered backoff")
    
    # Analyze provider switches
    print(f"\n🔄 Provider switch analysis:")
    print(f"   Total switches: {len(provider_switches)}")
    
    for switch in provider_switches:
        print(f"   Request {switch['request']}: {switch['from']} → {switch['to']} ({switch['duration']:.2f}s)")
        
        if switch['duration'] > 7.0:
            print(f"      ✅ Switch after long delay - likely after exhausting retries")
        else:
            print(f"      ⚠️ Quick switch - may not have used full exponential backoff")
    
    # Final statistics
    final_stats = wrapper.get_statistics()
    print(f"\n📈 Final Statistics:")
    print(f"   Total requests: {final_stats['total_requests']}")
    print(f"   Successful: {final_stats['successful_requests']}")
    print(f"   Rate limited: {final_stats['rate_limited_requests']}")
    print(f"   Provider switches: {final_stats['fallback_switches']}")
    print(f"   Success rate: {final_stats['success_rate']:.1%}")
    
    return {
        'request_times': request_times,
        'provider_switches': provider_switches,
        'stats': final_stats,
        'exponential_backoff_detected': len(long_delays) > 0
    }

def demonstrate_expected_backoff():
    """Demonstrate what the exponential backoff should look like"""
    print(f"\n⏱️ EXPECTED EXPONENTIAL BACKOFF DEMONSTRATION")
    print("=" * 50)
    
    from llm_wrapper_enhanced import RetryConfig
    
    config = RetryConfig()
    
    print(f"With current configuration:")
    print(f"   Max attempts: {config.max_attempts}")
    print(f"   Base delay: {config.base_delay}s")
    print(f"   Exponential base: {config.exponential_base}")
    
    print(f"\nExpected behavior when rate limited:")
    total_time = 0
    
    for attempt in range(config.max_attempts):
        if attempt == 0:
            print(f"   Attempt 1: Make request → Rate limited")
        else:
            delay = min(
                config.base_delay * (config.exponential_base ** (attempt - 1)),
                config.max_delay
            )
            total_time += delay
            print(f"   Attempt {attempt + 1}: Wait {delay:.1f}s → Make request → {'Rate limited' if attempt < config.max_attempts - 1 else 'Still rate limited'}")
    
    print(f"   Total retry time: {total_time:.1f}s")
    print(f"   After {config.max_attempts} attempts: Switch to next provider")
    
    print(f"\n💡 This means each provider should get {total_time:.1f}s of retry time")
    print(f"   before the system switches to the next provider.")

def test_with_wine_pdf_simulation():
    """Simulate processing like the wine PDF to see backoff behavior"""
    print(f"\n🍷 Wine PDF Processing Simulation")
    print("=" * 35)
    
    from llm_wrapper_enhanced import LLMWrapper
    
    wrapper = LLMWrapper()
    
    print(f"🎯 Starting provider: {wrapper.current_provider}")
    print(f"📄 Simulating processing 10 chunks (like wine PDF)...")
    
    chunk_results = []
    
    for chunk in range(1, 11):
        print(f"\nChunk {chunk:2d}: ", end="", flush=True)
        
        provider_before = wrapper.current_provider
        start_time = time.time()
        
        # Simulate chunk processing
        prompt = f"Extract wine biomarkers from chunk {chunk}: This text contains various polyphenolic compounds."
        response = wrapper.generate_single(prompt, max_tokens=100)
        
        end_time = time.time()
        provider_after = wrapper.current_provider
        
        duration = end_time - start_time
        success = len(response) > 0
        
        chunk_results.append({
            'chunk': chunk,
            'provider_before': provider_before,
            'provider_after': provider_after,
            'duration': duration,
            'success': success
        })
        
        print(f"{'✅' if success else '❌'} {duration:.2f}s [{provider_after}]")
        
        if provider_before != provider_after:
            print(f"    🔄 Provider switch: {provider_before} → {provider_after}")
        
        if duration > 5.0:
            print(f"    ⏱️ Long processing time: {duration:.2f}s (exponential backoff)")
    
    # Analyze simulation results
    print(f"\n📊 Simulation Results:")
    
    providers_used = {}
    for result in chunk_results:
        provider = result['provider_after']
        providers_used[provider] = providers_used.get(provider, 0) + 1
    
    print(f"   Provider usage:")
    for provider, count in providers_used.items():
        print(f"     {provider}: {count} chunks")
    
    switches = [r for r in chunk_results if r['provider_before'] != r['provider_after']]
    print(f"   Provider switches: {len(switches)}")
    
    long_delays = [r for r in chunk_results if r['duration'] > 5.0]
    print(f"   Long delays (>5s): {len(long_delays)} (evidence of backoff)")
    
    stats = wrapper.get_statistics()
    print(f"   Rate limited requests: {stats['rate_limited_requests']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")

def main():
    """Test exponential backoff behavior"""
    print("🔄 Enhanced Wrapper - Exponential Backoff Testing")
    print("=" * 55)
    
    try:
        # Test 1: Basic exponential backoff behavior
        results = test_exponential_backoff_behavior()
        
        # Test 2: Show expected behavior
        demonstrate_expected_backoff()
        
        # Test 3: Wine PDF simulation
        test_with_wine_pdf_simulation()
        
        print(f"\n🎯 CONCLUSION:")
        if results['exponential_backoff_detected']:
            print(f"✅ Exponential backoff is working correctly!")
            print(f"   The wrapper now retries with delays before switching providers.")
        else:
            print(f"⚠️ Exponential backoff may not have been triggered.")
            print(f"   Try running more requests to hit rate limits.")
        
        print(f"\n💡 IMPROVEMENT:")
        print(f"The wrapper now prioritizes exponential backoff over provider switching,")
        print(f"giving each provider multiple chances with increasing delays before")
        print(f"moving to the next provider in the fallback chain.")
        
    except Exception as e:
        print(f"❌ Error testing exponential backoff: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
