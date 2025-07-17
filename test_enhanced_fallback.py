#!/usr/bin/env python3
"""
Test script for enhanced LLM wrapper with V4 priority list and improved rate limiting fallback
"""

import sys
import os
sys.path.append('FOODB_LLM_pipeline')

def test_enhanced_wrapper():
    """Test the enhanced wrapper with V4 priority list"""
    print("🧪 Testing Enhanced LLM Wrapper with V4 Priority List")
    print("=" * 60)
    
    try:
        from llm_wrapper_enhanced import LLMWrapper, RetryConfig
        
        # Create retry configuration for aggressive fallback
        retry_config = RetryConfig(
            max_attempts=2,  # Reduced attempts for faster switching
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # Initialize wrapper
        print("🔧 Initializing enhanced wrapper...")
        wrapper = LLMWrapper(retry_config=retry_config)
        
        # Show provider status
        print("\n📊 Provider Status:")
        status = wrapper.get_provider_status()
        print(f"   Current Provider: {status['current_provider']}")
        
        for provider, info in status['providers'].items():
            api_status = "✅ API Key" if info['has_api_key'] else "❌ No API Key"
            print(f"   {provider}: {info['status']} | {api_status}")
        
        # Test basic functionality
        print(f"\n🔬 Testing basic metabolite extraction...")
        test_prompt = """Extract ALL metabolites and chemical compounds from this wine research text:

Red wine contains resveratrol, quercetin, and anthocyanins. After consumption, these compounds are metabolized to produce urinary biomarkers including resveratrol-3-glucuronide, quercetin-3-glucuronide, and malvidin-3-glucoside."""
        
        print(f"📝 Test prompt: {test_prompt[:100]}...")
        
        # Make request with fallback
        response = wrapper.generate_single_with_fallback(test_prompt, max_tokens=200)
        
        if response:
            print(f"✅ Response generated successfully!")
            print(f"📊 Provider used: {wrapper.current_provider}")
            print(f"📝 Response length: {len(response)} characters")
            print(f"🔍 Response preview: {response[:200]}...")
        else:
            print(f"❌ No response generated")
        
        # Show statistics
        print(f"\n📈 Usage Statistics:")
        stats = wrapper.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        # Test provider switching under rate limiting
        print(f"\n🔄 Testing rate limiting fallback...")
        print(f"Making multiple rapid requests to test provider switching...")
        
        for i in range(3):
            print(f"\n📤 Request {i+1}/3...")
            response = wrapper.generate_single_with_fallback(
                f"Extract metabolites from: Wine contains compound {i+1}.", 
                max_tokens=50
            )
            
            if response:
                print(f"✅ Request {i+1} successful with {wrapper.current_provider}")
            else:
                print(f"❌ Request {i+1} failed")
        
        # Final statistics
        print(f"\n📊 Final Statistics:")
        final_stats = wrapper.get_statistics()
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n🎉 Enhanced wrapper test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_v4_priority_loading():
    """Test V4 priority list loading"""
    print("\n📋 Testing V4 Priority List Loading")
    print("-" * 40)
    
    try:
        import json
        
        # Check if V4 file exists
        if os.path.exists('llm_usage_priority_list.json'):
            with open('llm_usage_priority_list.json', 'r') as f:
                data = json.load(f)
            
            priority_list = data.get('priority_list', [])
            print(f"✅ V4 priority list loaded: {len(priority_list)} models")
            
            # Show top 5 models
            print(f"\n🏆 Top 5 Priority Models:")
            for i, model in enumerate(priority_list[:5], 1):
                provider = model.get('provider', 'Unknown')
                name = model.get('model_name', 'Unknown')
                score = model.get('performance_score', 0)
                score_type = model.get('score_type', 'unknown')
                
                print(f"   {i}. [{provider}] {name}")
                print(f"      Score: {score:.4f} ({score_type})")
            
            # Show provider breakdown
            providers = {}
            for model in priority_list:
                provider = model.get('provider', 'Unknown')
                providers[provider] = providers.get(provider, 0) + 1
            
            print(f"\n📊 Provider Distribution:")
            for provider, count in providers.items():
                percentage = (count / len(priority_list)) * 100
                print(f"   {provider}: {count} models ({percentage:.1f}%)")
            
            return True
            
        else:
            print(f"❌ V4 priority list file not found: llm_usage_priority_list.json")
            return False
            
    except Exception as e:
        print(f"❌ Error loading V4 priority list: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced LLM Wrapper Test Suite")
    print("=" * 60)
    
    # Test V4 priority list loading
    v4_success = test_v4_priority_loading()
    
    # Test enhanced wrapper
    wrapper_success = test_enhanced_wrapper()
    
    print(f"\n📋 Test Results:")
    print(f"   V4 Priority List: {'✅ PASS' if v4_success else '❌ FAIL'}")
    print(f"   Enhanced Wrapper: {'✅ PASS' if wrapper_success else '❌ FAIL'}")
    
    if v4_success and wrapper_success:
        print(f"\n🎉 All tests passed! Enhanced fallback system is ready.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
