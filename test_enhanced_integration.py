#!/usr/bin/env python3
"""
Test Enhanced Wrapper Integration with FOODB Pipeline
"""

import sys
import os

# Add pipeline directory to path
sys.path.append('FOODB_LLM_pipeline')

def test_enhanced_wrapper_integration():
    """Test that enhanced wrapper works with pipeline"""
    print("🧪 Testing Enhanced Wrapper Integration")
    print("=" * 50)
    
    try:
        # Import enhanced wrapper
        from llm_wrapper_enhanced import LLMWrapper
        
        # Create wrapper
        wrapper = LLMWrapper()
        
        print(f"✅ Enhanced wrapper imported successfully")
        print(f"🎯 Primary provider: {wrapper.current_provider}")
        
        # Test basic functionality
        test_prompt = "Extract metabolites from: Red wine contains resveratrol."
        
        print(f"\n🔬 Testing basic functionality...")
        response = wrapper.generate_single(test_prompt, max_tokens=100)
        
        if response:
            print(f"✅ Response generated successfully")
            print(f"📝 Response length: {len(response)} characters")
            print(f"📊 Provider used: {wrapper.current_provider}")
        else:
            print(f"❌ No response generated")
        
        # Show statistics
        stats = wrapper.get_statistics()
        print(f"\n📈 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1%}" if 'rate' in key else f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_wrapper_integration()
