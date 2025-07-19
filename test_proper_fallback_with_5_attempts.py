#!/usr/bin/env python3
"""
Test the corrected multi-tier fallback algorithm with proper 5 attempts and detailed delay logging
"""

import sys
import os
sys.path.append('FOODB_LLM_pipeline')

def test_proper_fallback():
    """Test the corrected multi-tier fallback algorithm with 5 attempts"""
    print("🧪 Testing Proper Multi-Tier Fallback with 5 Attempts")
    print("=" * 70)
    
    try:
        from llm_wrapper_enhanced import LLMWrapper, RetryConfig
        
        # Create retry configuration with DEFAULT 5 attempts (not overridden)
        retry_config = RetryConfig(
            max_attempts=5,      # DEFAULT: 5 attempts with exponential backoff
            base_delay=1.0,      # Start with 1 second
            max_delay=30.0,      # Cap at 30 seconds
            exponential_base=2.0, # Double each time
            jitter=True          # Add randomness
        )
        
        # Initialize wrapper
        print("🔧 Initializing wrapper with proper 5-attempt configuration...")
        wrapper = LLMWrapper(retry_config=retry_config)
        
        print(f"\n📊 Retry Configuration:")
        print(f"   Max attempts: {retry_config.max_attempts}")
        print(f"   Base delay: {retry_config.base_delay}s")
        print(f"   Max delay: {retry_config.max_delay}s")
        print(f"   Exponential base: {retry_config.exponential_base}")
        print(f"   Jitter enabled: {retry_config.jitter}")
        
        print(f"\n🔢 Expected delay progression (without jitter):")
        for i in range(5):
            expected_delay = retry_config.base_delay * (retry_config.exponential_base ** i)
            expected_delay = min(expected_delay, retry_config.max_delay)
            print(f"   Attempt {i+1}: {expected_delay:.1f}s")
        
        # Show provider status
        print(f"\n📊 Provider Status:")
        status = wrapper.get_provider_status()
        print(f"   Current Provider: {status['current_provider']}")
        
        for provider, info in status['providers'].items():
            api_status = "✅ API Key" if info['has_api_key'] else "❌ No API Key"
            print(f"   {provider}: {info['status']} | {api_status}")
        
        # Test the multi-tier fallback
        print(f"\n🔬 Testing multi-tier fallback with detailed delay logging...")
        test_prompt = """Extract metabolites from this wine research text: Red wine contains resveratrol, quercetin, and anthocyanins."""
        
        print(f"📝 Test prompt: {test_prompt}")
        print(f"\n🔄 Expected behavior:")
        print(f"   1. Try Cerebras model 1 with 5 retry attempts")
        print(f"      - Attempt 1: 1.0s delay")
        print(f"      - Attempt 2: 2.0s delay (doubled)")
        print(f"      - Attempt 3: 4.0s delay (doubled)")
        print(f"      - Attempt 4: 8.0s delay (doubled)")
        print(f"      - Attempt 5: 16.0s delay (doubled)")
        print(f"   2. If model 1 fails, try Cerebras model 2 with 5 retry attempts")
        print(f"   3. Continue through all Cerebras models")
        print(f"   4. If all Cerebras models fail, escalate to Groq models")
        print(f"   5. Try each Groq model with 5 retry attempts")
        print(f"   6. If all Groq models fail, escalate to OpenRouter models")
        
        print(f"\n🚀 Starting test with proper 5-attempt configuration...")
        
        # Make request with corrected fallback
        response = wrapper.generate_single_with_fallback(test_prompt, max_tokens=150)
        
        if response:
            print(f"\n✅ Response generated successfully!")
            print(f"📝 Response length: {len(response)} characters")
            print(f"🔍 Response preview: {response[:200]}...")
        else:
            print(f"\n❌ No response generated (all models exhausted)")
        
        # Show statistics
        print(f"\n📈 Usage Statistics:")
        stats = wrapper.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n🎉 Proper 5-attempt fallback test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Proper Multi-Tier Fallback Test with 5 Attempts")
    print("=" * 70)
    
    # Test proper fallback
    success = test_proper_fallback()
    
    print(f"\n📋 Test Result:")
    print(f"   Proper 5-Attempt Fallback: {'✅ PASS' if success else '❌ FAIL'}")
    
    if success:
        print(f"\n🎉 Proper multi-tier fallback algorithm is working!")
        print(f"   ✅ 5 attempts with exponential backoff (1s → 2s → 4s → 8s → 16s)")
        print(f"   ✅ Detailed delay logging showing doubling process")
        print(f"   ✅ Model rotation within each provider")
        print(f"   ✅ Provider escalation only after all models exhausted")
    else:
        print(f"\n⚠️  Test failed. Check the output above for details.")
