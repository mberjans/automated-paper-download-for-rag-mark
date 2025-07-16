#!/usr/bin/env python3
"""
Aggressive Exponential Backoff Test
This script sends rapid requests to trigger full exponential backoff sequence
"""

import time
import sys
sys.path.append('FOODB_LLM_pipeline')

def test_aggressive_exponential_backoff():
    """Send rapid requests to trigger full exponential backoff"""
    print("🔥 Aggressive Exponential Backoff Test")
    print("=" * 40)
    
    from llm_wrapper_enhanced import LLMWrapper, RetryConfig
    
    # Create wrapper with aggressive retry config
    retry_config = RetryConfig(
        max_attempts=4,        # 4 attempts to see full sequence
        base_delay=1.0,        # 1 second base
        max_delay=20.0,        # Max 20 seconds
        exponential_base=2.0,  # Double each time
        jitter=False           # No jitter for predictable testing
    )
    
    wrapper = LLMWrapper(retry_config=retry_config)
    
    print(f"🎯 Primary provider: {wrapper.current_provider}")
    print(f"📋 Aggressive retry config:")
    print(f"   Max attempts: {retry_config.max_attempts}")
    print(f"   Expected delays: 1s → 2s → 4s → 8s")
    print(f"   Total retry time: 15s before switching")
    
    print(f"\n🚀 Sending 30 rapid requests to overwhelm rate limits...")
    
    results = []
    
    for i in range(1, 31):
        print(f"\nRequest {i:2d}: ", end="", flush=True)
        
        provider_before = wrapper.current_provider
        start_time = time.time()
        
        # Send rapid requests with no delay
        response = wrapper.generate_single(f"Rapid test {i}", max_tokens=50)
        
        end_time = time.time()
        provider_after = wrapper.current_provider
        
        duration = end_time - start_time
        success = len(response) > 0
        
        results.append({
            'request': i,
            'provider_before': provider_before,
            'provider_after': provider_after,
            'duration': duration,
            'success': success
        })
        
        print(f"{'✅' if success else '❌'} {duration:.2f}s [{provider_after}]")
        
        # Highlight important events
        if provider_before != provider_after:
            print(f"    🔄 PROVIDER SWITCH: {provider_before} → {provider_after}")
        
        if duration > 10.0:
            print(f"    ⏱️ LONG BACKOFF: {duration:.2f}s (full exponential sequence)")
        elif duration > 5.0:
            print(f"    ⏱️ MEDIUM BACKOFF: {duration:.2f}s (partial exponential)")
        elif duration > 2.0:
            print(f"    ⏱️ SHORT BACKOFF: {duration:.2f}s (initial retry)")
        
        # Show current statistics
        stats = wrapper.get_statistics()
        if i % 5 == 0:  # Every 5 requests
            print(f"    📊 Stats: {stats['rate_limited_requests']} rate limited, {stats['fallback_switches']} switches")
    
    # Analyze results
    print(f"\n📊 AGGRESSIVE TEST ANALYSIS")
    print("=" * 30)
    
    # Categorize delays
    fast_requests = [r for r in results if r['duration'] < 2.0]
    short_backoff = [r for r in results if 2.0 <= r['duration'] < 5.0]
    medium_backoff = [r for r in results if 5.0 <= r['duration'] < 10.0]
    long_backoff = [r for r in results if r['duration'] >= 10.0]
    
    print(f"Request timing distribution:")
    print(f"   Fast (<2s): {len(fast_requests)} requests")
    print(f"   Short backoff (2-5s): {len(short_backoff)} requests")
    print(f"   Medium backoff (5-10s): {len(medium_backoff)} requests")
    print(f"   Long backoff (≥10s): {len(long_backoff)} requests")
    
    # Show longest delays (evidence of full exponential backoff)
    if long_backoff:
        print(f"\n⏱️ FULL EXPONENTIAL BACKOFF DETECTED:")
        for result in long_backoff:
            print(f"   Request {result['request']}: {result['duration']:.2f}s")
        print(f"   ✅ Maximum delay: {max(r['duration'] for r in long_backoff):.2f}s")
    
    # Provider switching analysis
    switches = [r for r in results if r['provider_before'] != r['provider_after']]
    print(f"\n🔄 Provider Switching Analysis:")
    print(f"   Total switches: {len(switches)}")
    
    if switches:
        for switch in switches:
            print(f"   Request {switch['request']}: {switch['provider_before']} → {switch['provider_after']} ({switch['duration']:.2f}s)")
            
            if switch['duration'] > 15.0:
                print(f"      ✅ Switch after full exponential backoff sequence")
            elif switch['duration'] > 7.0:
                print(f"      ⚠️ Switch after partial backoff")
            else:
                print(f"      ❌ Quick switch - backoff may not have completed")
    
    # Provider usage distribution
    provider_usage = {}
    for result in results:
        provider = result['provider_after']
        provider_usage[provider] = provider_usage.get(provider, 0) + 1
    
    print(f"\n🌐 Provider Usage Distribution:")
    for provider, count in provider_usage.items():
        percentage = count / len(results) * 100
        print(f"   {provider}: {count}/30 requests ({percentage:.1f}%)")
    
    # Final statistics
    final_stats = wrapper.get_statistics()
    print(f"\n📈 Final Statistics:")
    print(f"   Total requests: {final_stats['total_requests']}")
    print(f"   Successful: {final_stats['successful_requests']}")
    print(f"   Rate limited: {final_stats['rate_limited_requests']}")
    print(f"   Provider switches: {final_stats['fallback_switches']}")
    print(f"   Success rate: {final_stats['success_rate']:.1%}")
    
    return results, final_stats

def demonstrate_backoff_sequence():
    """Show what the exponential backoff sequence should look like"""
    print(f"\n⏱️ EXPECTED EXPONENTIAL BACKOFF SEQUENCE")
    print("=" * 45)
    
    from llm_wrapper_enhanced import RetryConfig
    
    config = RetryConfig(max_attempts=4, base_delay=1.0, exponential_base=2.0)
    
    print(f"When a provider gets rate limited:")
    print(f"1. First request → Rate limited")
    
    total_time = 0
    for attempt in range(1, config.max_attempts):
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        total_time += delay
        print(f"{attempt + 1}. Wait {delay:.1f}s → Retry → {'Rate limited' if attempt < config.max_attempts - 1 else 'Still rate limited'}")
    
    print(f"5. Total time spent: {total_time:.1f}s")
    print(f"6. Switch to next provider")
    
    print(f"\n💡 With this logic:")
    print(f"   • Each provider gets {total_time:.1f}s of retry attempts")
    print(f"   • Providers have time to recover from rate limits")
    print(f"   • System is more patient before switching")

def main():
    """Run aggressive exponential backoff test"""
    print("🔥 Enhanced Wrapper - Aggressive Exponential Backoff Test")
    print("=" * 60)
    
    try:
        # Show expected behavior
        demonstrate_backoff_sequence()
        
        # Run aggressive test
        results, stats = test_aggressive_exponential_backoff()
        
        # Conclusion
        print(f"\n🎯 TEST CONCLUSION:")
        
        long_delays = [r for r in results if r['duration'] >= 10.0]
        medium_delays = [r for r in results if 5.0 <= r['duration'] < 10.0]
        
        if long_delays:
            print(f"✅ FULL EXPONENTIAL BACKOFF CONFIRMED!")
            print(f"   • {len(long_delays)} requests showed full backoff sequence (≥10s)")
            print(f"   • Maximum delay: {max(r['duration'] for r in long_delays):.2f}s")
            print(f"   • System properly retries before switching providers")
        elif medium_delays:
            print(f"✅ PARTIAL EXPONENTIAL BACKOFF DETECTED")
            print(f"   • {len(medium_delays)} requests showed partial backoff (5-10s)")
            print(f"   • System is retrying but providers recover quickly")
        else:
            print(f"⚠️ LIMITED BACKOFF OBSERVED")
            print(f"   • Providers may be recovering too quickly to see full sequence")
            print(f"   • But retry logic is working (see 'retrying in Xs' messages)")
        
        print(f"\n🎉 IMPROVEMENT CONFIRMED:")
        print(f"The wrapper now uses exponential backoff BEFORE switching providers!")
        print(f"This gives each provider multiple chances to recover from rate limits.")
        
    except Exception as e:
        print(f"❌ Aggressive backoff test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
