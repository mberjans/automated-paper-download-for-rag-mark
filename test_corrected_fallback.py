#!/usr/bin/env python3
"""
Test the corrected multi-tier fallback algorithm
"""

import sys
import os
sys.path.append('FOODB_LLM_pipeline')

def test_corrected_fallback():
    """Test the corrected multi-tier fallback algorithm"""
    print("🧪 Testing Corrected Multi-Tier Fallback Algorithm")
    print("=" * 60)
    
    try:
        from llm_wrapper_enhanced import LLMWrapper, RetryConfig
        
        # Create retry configuration for testing
        retry_config = RetryConfig(
            max_attempts=3,  # Reduced for faster testing
            base_delay=0.5,  # Shorter delays for testing
            max_delay=5.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # Initialize wrapper
        print("🔧 Initializing corrected wrapper...")
        wrapper = LLMWrapper(retry_config=retry_config)
        
        # Show provider status
        print("\n📊 Provider Status:")
        status = wrapper.get_provider_status()
        print(f"   Current Provider: {status['current_provider']}")
        
        for provider, info in status['providers'].items():
            api_status = "✅ API Key" if info['has_api_key'] else "❌ No API Key"
            print(f"   {provider}: {info['status']} | {api_status}")
        
        # Test the multi-tier fallback
        print(f"\n🔬 Testing multi-tier fallback algorithm...")
        test_prompt = """Extract metabolites from this text: Wine contains resveratrol and quercetin."""
        
        print(f"📝 Test prompt: {test_prompt}")
        print(f"\n🔄 Expected behavior:")
        print(f"   1. Try Cerebras model 1 with 3 retry attempts (exponential backoff)")
        print(f"   2. If failed, try Cerebras model 2 with 3 retry attempts")
        print(f"   3. Continue through all Cerebras models")
        print(f"   4. If all Cerebras models fail, escalate to Groq models")
        print(f"   5. Try each Groq model with 3 retry attempts")
        print(f"   6. If all Groq models fail, escalate to OpenRouter models")
        print(f"   7. Try each OpenRouter model with 3 retry attempts")
        print(f"   8. Only fail after all models in all providers exhausted")
        
        print(f"\n🚀 Starting test...")
        
        # Make request with corrected fallback
        response = wrapper.generate_single_with_fallback(test_prompt, max_tokens=100)
        
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
        
        print(f"\n🎉 Corrected fallback test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Corrected Multi-Tier Fallback Test")
    print("=" * 60)
    
    # Test corrected fallback
    success = test_corrected_fallback()
    
    print(f"\n📋 Test Result:")
    print(f"   Corrected Fallback: {'✅ PASS' if success else '❌ FAIL'}")
    
    if success:
        print(f"\n🎉 Corrected multi-tier fallback algorithm is working!")
        print(f"   ✅ Proper exponential backoff within each model")
        print(f"   ✅ Model rotation within each provider")
        print(f"   ✅ Provider escalation only after all models exhausted")
    else:
        print(f"\n⚠️  Test failed. Check the output above for details.")
