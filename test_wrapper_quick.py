#!/usr/bin/env python3
"""
Quick test of the FOODB LLM Pipeline Wrapper
This script tests basic functionality with available API keys
"""

import sys
import os

# Add the FOODB_LLM_pipeline directory to the path
sys.path.append('FOODB_LLM_pipeline')

def test_basic_functionality():
    """Test basic wrapper functionality"""
    print("🧪 Testing FOODB LLM Pipeline Wrapper")
    print("=" * 50)
    
    try:
        from llm_wrapper import LLMWrapper
        
        # Create wrapper
        wrapper = LLMWrapper()
        
        # Check available models
        models = wrapper.list_available_models()
        print(f"📋 Found {len(models)} model configurations")
        
        if not models:
            print("❌ No model configurations found")
            return False
        
        # Try to set a model (test API connectivity)
        print("\n🔌 Testing API connectivity...")
        
        # Test Cerebras first (best performance)
        cerebras_models = [m for m in models if m.get('provider', '').lower() == 'cerebras']
        if cerebras_models:
            print("Testing Cerebras models...")
            for model in cerebras_models[:2]:  # Test first 2
                try:
                    if wrapper.set_model(model):
                        response = wrapper.generate_single("Say 'Hello from FOODB!'", max_tokens=10)
                        if response.strip():
                            print(f"✅ {model.get('model_name')} working: {response.strip()}")
                            return True
                        else:
                            print(f"❌ {model.get('model_name')} - No response")
                    else:
                        print(f"❌ {model.get('model_name')} - Failed to initialize")
                except Exception as e:
                    print(f"❌ {model.get('model_name')} - Error: {e}")
        
        # Test OpenRouter as fallback
        openrouter_models = [m for m in models if m.get('provider', '').lower() == 'openrouter']
        if openrouter_models:
            print("Testing OpenRouter models...")
            for model in openrouter_models[:2]:  # Test first 2
                try:
                    if wrapper.set_model(model):
                        response = wrapper.generate_single("Say 'Hello from FOODB!'", max_tokens=10)
                        if response.strip():
                            print(f"✅ {model.get('model_name')} working: {response.strip()}")
                            return True
                        else:
                            print(f"❌ {model.get('model_name')} - No response")
                    else:
                        print(f"❌ {model.get('model_name')} - Failed to initialize")
                except Exception as e:
                    print(f"❌ {model.get('model_name')} - Error: {e}")
        
        print("❌ No working models found")
        return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory and have the required dependencies")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_scientific_text_processing():
    """Test processing scientific text"""
    print("\n🔬 Testing Scientific Text Processing")
    print("=" * 50)
    
    try:
        from llm_wrapper import LLMWrapper
        
        wrapper = LLMWrapper()
        
        # Try to set any working model
        models = wrapper.list_available_models()
        working_model = None
        
        for model in models[:5]:  # Test first 5 models
            try:
                if wrapper.set_model(model):
                    test_response = wrapper.generate_single("Test", max_tokens=5)
                    if test_response.strip():
                        working_model = model
                        break
            except:
                continue
        
        if not working_model:
            print("❌ No working models found for scientific text processing")
            return False
        
        print(f"✅ Using model: {working_model.get('model_name')}")
        
        # Test scientific text processing
        scientific_text = """Resveratrol is a polyphenolic compound found in red wine that exhibits various beneficial effects on metabolic health. Studies have shown that resveratrol activates SIRT1, a protein that plays a key role in cellular stress resistance and longevity."""
        
        print(f"\n📝 Input text: {scientific_text[:100]}...")
        
        # Test simple sentence generation
        prompt = f"""Convert the given scientific text into a list of simple, clear sentences. Each sentence should express a single fact or relationship.

Text: {scientific_text}

Simple sentences:"""
        
        response = wrapper.generate_single(prompt, max_tokens=200)
        
        if response.strip():
            print(f"\n✅ Generated response:")
            print(response.strip())
            return True
        else:
            print("❌ No response generated")
            return False
            
    except Exception as e:
        print(f"❌ Error in scientific text processing: {e}")
        return False


def main():
    """Main test function"""
    print("🧬 FOODB LLM Pipeline Wrapper - Quick Test")
    print("=" * 60)
    
    # Test 1: Basic functionality
    basic_test_passed = test_basic_functionality()
    
    if not basic_test_passed:
        print("\n❌ Basic functionality test failed")
        print("\n💡 Troubleshooting tips:")
        print("1. Check that your API keys are set in .env file")
        print("2. Verify internet connectivity")
        print("3. Make sure you have the required dependencies:")
        print("   pip install requests openai")
        return
    
    # Test 2: Scientific text processing
    scientific_test_passed = test_scientific_text_processing()
    
    if scientific_test_passed:
        print("\n✅ All tests passed! The FOODB LLM Pipeline Wrapper is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Try the full test suite: python FOODB_LLM_pipeline/test_wrapper.py")
        print("2. Run examples: python FOODB_LLM_pipeline/example_usage.py")
        print("3. Use the wrapper in your own scripts")
    else:
        print("\n⚠️ Basic connectivity works, but scientific text processing failed")
        print("The wrapper is partially functional")


if __name__ == "__main__":
    main()
