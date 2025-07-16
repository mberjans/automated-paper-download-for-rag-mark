#!/usr/bin/env python3
"""
Test Cerebras Rate Limiting and Fallback Behavior
This script aggressively tests Cerebras to trigger rate limiting and demonstrate fallback
"""

import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from enhanced_llm_wrapper_with_fallback import EnhancedLLMWrapper, RetryConfig

def test_cerebras_aggressive_rate_limiting():
    """Aggressively test Cerebras to trigger rate limiting"""
    print("🔥 Testing Cerebras Aggressive Rate Limiting")
    print("=" * 50)
    
    # Configure for aggressive testing
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay=0.5,  # Short delays
        max_delay=10.0,
        exponential_base=2.0,
        jitter=False
    )
    
    wrapper = EnhancedLLMWrapper(retry_config=retry_config)
    
    print(f"🎯 Starting provider: {wrapper.current_provider}")
    
    if wrapper.current_provider != 'cerebras':
        print(f"❌ Expected Cerebras, got {wrapper.current_provider}")
        return
    
    # Create 30 rapid requests to trigger Cerebras rate limiting
    print(f"\n🚀 Sending 30 rapid requests to trigger Cerebras rate limiting...")
    
    results = []
    start_time = time.time()
    
    for i in range(1, 31):
        prompt = f"Extract metabolites from sample {i}: Wine contains various polyphenolic compounds including resveratrol and anthocyanins."
        
        print(f"Request {i:2d}: ", end="", flush=True)
        
        provider_before = wrapper.get_provider_status()['current_provider']
        
        request_start = time.time()
        response = wrapper.generate_single_with_fallback(prompt, max_tokens=100)
        request_end = time.time()
        
        provider_after = wrapper.get_provider_status()['current_provider']
        success = len(response) > 0
        
        results.append({
            'request': i,
            'provider_before': provider_before,
            'provider_after': provider_after,
            'success': success,
            'time': request_end - request_start,
            'response_length': len(response)
        })
        
        # Show result with provider info
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {request_end - request_start:.2f}s [{provider_after}]")
        
        # Highlight provider switches
        if provider_before != provider_after:
            print(f"    🔄 PROVIDER SWITCH: {provider_before} → {provider_after}")
        
        # Show stats every 10 requests
        if i % 10 == 0:
            stats = wrapper.get_statistics()
            print(f"    📊 {i}/30: Success {stats['successful_requests']}/{stats['total_requests']}, "
                  f"Rate Limited: {stats['rate_limited_requests']}, "
                  f"Switches: {stats['fallback_switches']}")
        
        # NO DELAY - maximum pressure on Cerebras
    
    total_time = time.time() - start_time
    
    # Analyze results
    print(f"\n📊 CEREBRAS RATE LIMITING RESULTS")
    print("=" * 45)
    print(f"Total requests: 30")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests per second: {30/total_time:.1f}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Success rate: {successful}/30 ({successful/30:.1%})")
    
    # Analyze provider usage
    provider_usage = {}
    for result in results:
        provider = result['provider_after']
        provider_usage[provider] = provider_usage.get(provider, 0) + 1
    
    print(f"\n🌐 Provider Usage:")
    for provider, count in provider_usage.items():
        percentage = count / 30 * 100
        print(f"  {provider}: {count}/30 ({percentage:.1f}%)")
    
    # Find when rate limiting started
    switches = [r for r in results if r['provider_before'] != r['provider_after']]
    if switches:
        first_switch = switches[0]
        print(f"\n🚨 Rate Limiting Analysis:")
        print(f"  First provider switch at request: {first_switch['request']}")
        print(f"  Switch: {first_switch['provider_before']} → {first_switch['provider_after']}")
        print(f"  Rate limiting started at: {first_switch['request']/30*100:.1f}% through test")
        
        print(f"\n🔄 All Provider Switches:")
        for switch in switches:
            print(f"    Request {switch['request']:2d}: {switch['provider_before']} → {switch['provider_after']}")
    else:
        print(f"\n✅ No rate limiting detected - Cerebras handled all requests")
    
    # Final statistics
    stats = wrapper.get_statistics()
    print(f"\n📈 Final Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            if 'rate' in key:
                print(f"  {key}: {value:.1%}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Provider health after test
    print(f"\n🏥 Provider Health After Test:")
    status = wrapper.get_provider_status()
    for provider, info in status['providers'].items():
        print(f"  {provider}: {info['status']} (failures: {info['consecutive_failures']})")
    
    return results, stats

def test_concurrent_cerebras_bombardment():
    """Test concurrent requests to Cerebras"""
    print(f"\n💥 Testing Concurrent Cerebras Bombardment")
    print("=" * 45)
    
    wrapper = EnhancedLLMWrapper()
    
    if wrapper.current_provider != 'cerebras':
        print(f"❌ Expected Cerebras, got {wrapper.current_provider}")
        return
    
    def make_concurrent_request(request_id):
        """Make a concurrent request"""
        prompt = f"Extract compounds from concurrent sample {request_id}: Contains metabolites."
        start_time = time.time()
        
        try:
            response = wrapper.generate_single_with_fallback(prompt, max_tokens=50)
            end_time = time.time()
            
            return {
                'id': request_id,
                'success': len(response) > 0,
                'time': end_time - start_time,
                'provider': wrapper.get_provider_status()['current_provider'],
                'response_length': len(response)
            }
        except Exception as e:
            end_time = time.time()
            return {
                'id': request_id,
                'success': False,
                'time': end_time - start_time,
                'provider': None,
                'error': str(e)
            }
    
    # Launch 15 concurrent requests
    print("🚀 Launching 15 concurrent requests to Cerebras...")
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(make_concurrent_request, i): i for i in range(1, 16)}
        
        concurrent_results = []
        for future in as_completed(futures):
            result = future.result()
            concurrent_results.append(result)
            
            status = "✅" if result['success'] else "❌"
            provider = result.get('provider', 'unknown')
            print(f"  Request {result['id']:2d}: {status} {result['time']:.2f}s [{provider}]")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Sort results by ID
    concurrent_results.sort(key=lambda x: x['id'])
    
    print(f"\n📊 Concurrent Bombardment Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Effective RPS: {15/total_time:.1f}")
    
    successful = sum(1 for r in concurrent_results if r['success'])
    print(f"Success rate: {successful}/15 ({successful/15:.1%})")
    
    # Provider distribution
    providers_used = {}
    for result in concurrent_results:
        if result['success'] and result['provider']:
            providers_used[result['provider']] = providers_used.get(result['provider'], 0) + 1
    
    print(f"\n🌐 Provider Distribution:")
    for provider, count in providers_used.items():
        print(f"  {provider}: {count} requests")
    
    # Check if fallback occurred
    stats = wrapper.get_statistics()
    if stats['fallback_switches'] > 0:
        print(f"\n🔄 Fallback occurred: {stats['fallback_switches']} switches")
        print(f"Rate limited requests: {stats['rate_limited_requests']}")
    else:
        print(f"\n✅ No fallback needed - Cerebras handled all concurrent requests")
    
    return concurrent_results

def save_cerebras_test_results(sequential_results, stats, concurrent_results):
    """Save test results"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"cerebras_rate_limiting_test_{timestamp}.json"
    
    data = {
        'timestamp': timestamp,
        'test_type': 'cerebras_rate_limiting',
        'sequential_test': {
            'results': sequential_results,
            'statistics': stats
        },
        'concurrent_test': concurrent_results,
        'summary': {
            'sequential_requests': len(sequential_results),
            'concurrent_requests': len(concurrent_results),
            'total_rate_limited': stats['rate_limited_requests'],
            'total_fallback_switches': stats['fallback_switches'],
            'cerebras_performance': 'excellent' if stats['fallback_switches'] == 0 else 'triggered_fallback'
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Cerebras test results saved to: {filename}")
    return filename

def main():
    """Run comprehensive Cerebras rate limiting tests"""
    print("🧬 FOODB Enhanced Wrapper - Cerebras Rate Limiting Test")
    print("=" * 60)
    
    try:
        # Test 1: Aggressive sequential requests
        sequential_results, stats = test_cerebras_aggressive_rate_limiting()
        
        # Brief pause
        time.sleep(2)
        
        # Test 2: Concurrent bombardment
        concurrent_results = test_concurrent_cerebras_bombardment()
        
        # Save results
        save_cerebras_test_results(sequential_results, stats, concurrent_results)
        
        print(f"\n🎉 Cerebras Rate Limiting Tests Completed!")
        
        if stats['fallback_switches'] > 0:
            print(f"✅ SUCCESS: Rate limiting triggered and fallback worked!")
            print(f"  🚨 Cerebras rate limited after {stats['rate_limited_requests']} requests")
            print(f"  🔄 System switched to fallback providers {stats['fallback_switches']} times")
            print(f"  ✅ Overall success rate: {stats['success_rate']:.1%}")
        else:
            print(f"📊 RESULT: Cerebras handled all requests without rate limiting")
            print(f"  ✅ Success rate: {stats['success_rate']:.1%}")
            print(f"  💪 Cerebras has very generous rate limits")
        
        print(f"\nThe enhanced wrapper demonstrates:")
        print(f"  🛡️ Robust error handling")
        print(f"  🔄 Intelligent provider switching")
        print(f"  ⚡ Fast fallback detection")
        print(f"  📊 Comprehensive monitoring")
        
    except Exception as e:
        print(f"❌ Cerebras rate limiting test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
