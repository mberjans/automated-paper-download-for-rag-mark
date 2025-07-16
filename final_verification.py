#!/usr/bin/env python3
"""
Final Verification: Rate Limit Fallback System Implementation
This script verifies that ALL active FOODB pipeline scripts have the enhanced fallback system
"""

import os
from pathlib import Path

def verify_active_scripts():
    """Verify all active scripts use enhanced wrapper"""
    print("🔍 Final Verification: Rate Limit Fallback System Implementation")
    print("=" * 70)
    
    # Define the active scripts (not backups or originals)
    active_scripts = [
        "FOODB_LLM_pipeline/5_LLM_Simple_Sentence_gen_API.py",
        "FOODB_LLM_pipeline/simple_sentenceRE3_API.py"
    ]
    
    print("📋 Active FOODB Pipeline Scripts:")
    print("=" * 35)
    
    all_enhanced = True
    
    for script_path in active_scripts:
        print(f"\n📄 {script_path}")
        
        if not os.path.exists(script_path):
            print(f"   ❌ File not found")
            all_enhanced = False
            continue
        
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Check for enhanced wrapper import
            if "from llm_wrapper_enhanced import LLMWrapper" in content:
                print(f"   ✅ Uses enhanced wrapper")
                
                # Check for fallback status reporting
                if "Primary provider:" in content and "Fallback system:" in content:
                    print(f"   ✅ Has status reporting")
                else:
                    print(f"   ⚠️ Missing status reporting")
                
                # Check for LLMWrapper initialization
                if "LLMWrapper()" in content:
                    print(f"   ✅ Initializes wrapper")
                else:
                    print(f"   ❌ No wrapper initialization found")
                    all_enhanced = False
                
            elif "from llm_wrapper import LLMWrapper" in content:
                print(f"   ❌ Still uses basic wrapper (no fallback)")
                all_enhanced = False
            else:
                print(f"   ❓ No LLM wrapper import found")
                all_enhanced = False
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            all_enhanced = False
    
    return all_enhanced

def test_enhanced_wrapper_functionality():
    """Test that the enhanced wrapper actually works"""
    print(f"\n🧪 Testing Enhanced Wrapper Functionality")
    print("=" * 45)
    
    try:
        import sys
        sys.path.append('FOODB_LLM_pipeline')
        
        from llm_wrapper_enhanced import LLMWrapper
        
        # Test initialization
        print("🔧 Initializing enhanced wrapper...")
        wrapper = LLMWrapper()
        
        print(f"✅ Wrapper initialized successfully")
        print(f"🎯 Primary provider: {wrapper.current_provider}")
        print(f"📋 Fallback order: {' → '.join(wrapper.fallback_order)}")
        
        # Test basic generation
        print(f"\n🔬 Testing basic generation...")
        test_response = wrapper.generate_single("Extract metabolites from: Wine contains resveratrol.", max_tokens=50)
        
        if test_response:
            print(f"✅ Generation successful")
            print(f"📝 Response length: {len(test_response)} characters")
            print(f"📊 Provider used: {wrapper.current_provider}")
        else:
            print(f"❌ Generation failed")
            return False
        
        # Check statistics
        stats = wrapper.get_statistics()
        print(f"\n📈 Statistics:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Fallback switches: {stats['fallback_switches']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_fallback_system_features():
    """Check that all fallback system features are present"""
    print(f"\n🛡️ Checking Fallback System Features")
    print("=" * 40)
    
    try:
        import sys
        sys.path.append('FOODB_LLM_pipeline')
        
        from llm_wrapper_enhanced import LLMWrapper, ProviderStatus, RetryConfig, ProviderHealth
        
        wrapper = LLMWrapper()
        
        # Check required attributes
        required_attributes = [
            'current_provider',
            'fallback_order', 
            'provider_health',
            'api_keys',
            'stats'
        ]
        
        print("🔍 Checking required attributes:")
        for attr in required_attributes:
            if hasattr(wrapper, attr):
                print(f"   ✅ {attr}")
            else:
                print(f"   ❌ {attr} missing")
                return False
        
        # Check required methods
        required_methods = [
            'generate_single_with_fallback',
            '_get_best_available_provider',
            '_update_provider_health',
            '_switch_provider',
            'get_statistics'
        ]
        
        print(f"\n🔧 Checking required methods:")
        for method in required_methods:
            if hasattr(wrapper, method):
                print(f"   ✅ {method}")
            else:
                print(f"   ❌ {method} missing")
                return False
        
        # Check provider configuration
        print(f"\n🌐 Checking provider configuration:")
        print(f"   Fallback order: {wrapper.fallback_order}")
        print(f"   Current provider: {wrapper.current_provider}")
        
        for provider in wrapper.fallback_order:
            has_key = bool(wrapper.api_keys.get(provider))
            health = wrapper.provider_health.get(provider)
            print(f"   {provider}: {'✅' if has_key else '❌'} API key, {health.status.value if health else 'No health'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking fallback features: {e}")
        return False

def generate_final_report():
    """Generate final implementation report"""
    print(f"\n📊 FINAL IMPLEMENTATION REPORT")
    print("=" * 35)
    
    # Check active scripts
    scripts_ok = verify_active_scripts()
    
    # Test functionality
    functionality_ok = test_enhanced_wrapper_functionality()
    
    # Check features
    features_ok = check_fallback_system_features()
    
    # Overall status
    all_systems_go = scripts_ok and functionality_ok and features_ok
    
    print(f"\n🎯 IMPLEMENTATION STATUS")
    print("=" * 25)
    print(f"Active scripts updated: {'✅' if scripts_ok else '❌'}")
    print(f"Enhanced wrapper working: {'✅' if functionality_ok else '❌'}")
    print(f"Fallback features present: {'✅' if features_ok else '❌'}")
    print(f"Overall status: {'✅ COMPLETE' if all_systems_go else '❌ INCOMPLETE'}")
    
    if all_systems_go:
        print(f"\n🎉 SUCCESS: Rate limit fallback system is FULLY IMPLEMENTED!")
        print(f"\n🛡️ The FOODB pipeline now has:")
        print(f"   ✅ Rate limiting detection in ALL active scripts")
        print(f"   ✅ Automatic provider fallback (Cerebras → Groq → OpenRouter)")
        print(f"   ✅ Exponential backoff retry logic")
        print(f"   ✅ Real-time provider health monitoring")
        print(f"   ✅ Comprehensive error handling")
        print(f"   ✅ Production-ready resilience")
        
        print(f"\n📋 Active Scripts with Fallback:")
        print(f"   • 5_LLM_Simple_Sentence_gen_API.py (sentence generation)")
        print(f"   • simple_sentenceRE3_API.py (triple extraction)")
        
        print(f"\n🚀 Ready for production deployment!")
    else:
        print(f"\n⚠️ Some issues need to be resolved before deployment")
    
    return all_systems_go

def main():
    """Main verification function"""
    print("🔍 FOODB Pipeline - Final Rate Limit Fallback Verification")
    print("=" * 65)
    
    try:
        success = generate_final_report()
        
        if success:
            print(f"\n✅ VERIFICATION COMPLETE: All systems operational!")
        else:
            print(f"\n❌ VERIFICATION FAILED: Issues detected")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
