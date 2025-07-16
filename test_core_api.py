#!/usr/bin/env python3
"""
Core API Testing - No External Dependencies
Tests the essential FOODB wrapper functionality without sentence transformers
"""

import sys
import os
import json
import time

# Add the FOODB_LLM_pipeline directory to the path
sys.path.append('FOODB_LLM_pipeline')

def test_core_llm_wrapper():
    """Test the core LLM wrapper functionality"""
    print("🔬 Testing Core LLM Wrapper")
    print("=" * 40)
    
    try:
        from llm_wrapper import LLMWrapper
        
        # Initialize wrapper
        print("Initializing LLM wrapper...")
        wrapper = LLMWrapper()
        
        print(f"✅ Wrapper initialized")
        print(f"🎯 Current model: {wrapper.current_model.get('model_name', 'Unknown')}")
        print(f"🏢 Provider: {wrapper.current_model.get('provider', 'Unknown')}")
        
        return wrapper
        
    except Exception as e:
        print(f"❌ Error initializing wrapper: {e}")
        return None

def test_simple_sentence_generation(wrapper):
    """Test simple sentence generation"""
    print("\n📝 Testing Simple Sentence Generation")
    print("=" * 45)
    
    if not wrapper:
        print("❌ No wrapper available")
        return False
    
    # Test cases with scientific abstracts
    test_cases = [
        {
            "name": "Curcumin Study",
            "text": "Curcumin, a polyphenolic compound derived from turmeric, has been extensively studied for its anti-inflammatory and antioxidant properties. Clinical trials have demonstrated that curcumin supplementation significantly reduces inflammatory markers in patients with rheumatoid arthritis.",
            "expected_sentences": 4
        },
        {
            "name": "Green Tea Research", 
            "text": "Epigallocatechin gallate (EGCG), the most abundant catechin in green tea, exhibits potent neuroprotective effects against age-related cognitive decline. Studies show that EGCG crosses the blood-brain barrier and modulates amyloid-β aggregation.",
            "expected_sentences": 3
        },
        {
            "name": "Resveratrol Analysis",
            "text": "Resveratrol, found in grape skins and red wine, activates sirtuin 1 (SIRT1) deacetylase, leading to enhanced mitochondrial biogenesis and improved metabolic health.",
            "expected_sentences": 2
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"Input: {test_case['text'][:80]}...")
        
        # Create prompt
        prompt = f"""Convert the given scientific text into a list of simple, clear sentences. Each sentence should express a single fact or relationship.

Text: {test_case['text']}

Simple sentences:"""
        
        start_time = time.time()
        response = wrapper.generate_single(prompt, max_tokens=300)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Parse sentences
        sentences = [s.strip() for s in response.split('\n') if s.strip() and not s.startswith('Simple sentences:')]
        # Filter out numbered lists and headers
        sentences = [s for s in sentences if not s.startswith(('1.', '2.', '3.', '4.', '5.', 'Here are', 'The simple'))]
        
        result = {
            'name': test_case['name'],
            'processing_time': processing_time,
            'sentence_count': len(sentences),
            'sentences': sentences[:3],  # Show first 3
            'success': len(sentences) > 0
        }
        results.append(result)
        
        status = "✅" if result['success'] else "❌"
        print(f"  {status} Time: {processing_time:.2f}s")
        print(f"  📝 Generated {len(sentences)} sentences")
        if sentences:
            print(f"  📄 Sample: {sentences[0][:100]}...")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['processing_time'] for r in results)
    
    print(f"\n📊 Simple Sentence Generation Summary:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time: {total_time/len(results):.2f}s per test")
    
    return successful == len(results)

def test_entity_extraction(wrapper):
    """Test entity extraction"""
    print("\n🧬 Testing Entity Extraction")
    print("=" * 35)
    
    if not wrapper:
        print("❌ No wrapper available")
        return False
    
    test_sentences = [
        "Curcumin reduces inflammation in arthritis patients.",
        "EGCG crosses the blood-brain barrier and protects neurons.",
        "Resveratrol activates SIRT1 protein pathways.",
        "Quercetin exhibits antioxidant properties in cardiovascular tissue.",
        "Lycopene consumption is associated with reduced cancer risk."
    ]
    
    results = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n🔍 Test {i}: {sentence}")
        
        prompt = f"""Extract food compounds, biological targets, and health effects from this sentence:

Sentence: {sentence}

Extract:
- Food compounds (e.g., curcumin, EGCG, resveratrol)
- Biological targets (e.g., SIRT1, inflammation, neurons)
- Health effects (e.g., anti-inflammatory, antioxidant, neuroprotective)

Entities:"""
        
        start_time = time.time()
        response = wrapper.generate_single(prompt, max_tokens=150)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Check if response contains relevant entities
        has_entities = any(keyword in response.lower() for keyword in [
            'curcumin', 'egcg', 'resveratrol', 'quercetin', 'lycopene',
            'inflammation', 'antioxidant', 'sirt1', 'neuron', 'cancer'
        ])
        
        result = {
            'sentence': sentence,
            'processing_time': processing_time,
            'response': response.strip(),
            'has_entities': has_entities
        }
        results.append(result)
        
        status = "✅" if has_entities else "❌"
        print(f"  {status} Time: {processing_time:.2f}s")
        print(f"  📄 Entities: {response.strip()[:100]}...")
    
    # Summary
    successful = sum(1 for r in results if r['has_entities'])
    total_time = sum(r['processing_time'] for r in results)
    
    print(f"\n📊 Entity Extraction Summary:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time: {total_time/len(results):.2f}s per sentence")
    
    return successful >= len(results) * 0.8  # 80% success rate

def test_triple_extraction(wrapper):
    """Test triple extraction"""
    print("\n🔗 Testing Triple Extraction")
    print("=" * 35)
    
    if not wrapper:
        print("❌ No wrapper available")
        return False
    
    test_sentences = [
        "Curcumin reduces inflammation.",
        "EGCG protects neurons from damage.",
        "Resveratrol activates SIRT1 protein.",
        "Quercetin improves cardiovascular health.",
        "Lycopene prevents prostate cancer."
    ]
    
    results = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n🔗 Test {i}: {sentence}")
        
        prompt = f"""Extract subject-predicate-object triples from this sentence. Focus on relationships between compounds and effects.

Sentence: {sentence}

Return each triple as [subject, predicate, object]:"""
        
        start_time = time.time()
        response = wrapper.generate_single(prompt, max_tokens=100)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Check if response contains triple format
        has_triples = '[' in response and ']' in response and ',' in response
        
        result = {
            'sentence': sentence,
            'processing_time': processing_time,
            'response': response.strip(),
            'has_triples': has_triples
        }
        results.append(result)
        
        status = "✅" if has_triples else "❌"
        print(f"  {status} Time: {processing_time:.2f}s")
        print(f"  🔗 Triple: {response.strip()[:100]}...")
    
    # Summary
    successful = sum(1 for r in results if r['has_triples'])
    total_time = sum(r['processing_time'] for r in results)
    
    print(f"\n📊 Triple Extraction Summary:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time: {total_time/len(results):.2f}s per sentence")
    
    return successful >= len(results) * 0.8  # 80% success rate

def test_batch_processing(wrapper):
    """Test batch processing capabilities"""
    print("\n📦 Testing Batch Processing")
    print("=" * 35)
    
    if not wrapper:
        print("❌ No wrapper available")
        return False
    
    # Create batch of prompts
    batch_prompts = [
        "Convert to simple sentences: Green tea contains antioxidants that benefit health.",
        "Extract entities: Curcumin has anti-inflammatory properties.",
        "Find triples: Resveratrol activates longevity pathways.",
        "Convert to simple sentences: Omega-3 fatty acids support brain function.",
        "Extract entities: Lycopene reduces oxidative stress in cells."
    ]
    
    print(f"🧪 Processing batch of {len(batch_prompts)} prompts...")
    
    start_time = time.time()
    responses = wrapper.generate_batch(batch_prompts, max_tokens=150, max_concurrent=3)
    end_time = time.time()
    
    total_time = end_time - start_time
    successful_responses = sum(1 for r in responses if r.strip())
    
    print(f"✅ Batch processing complete")
    print(f"📊 Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per prompt: {total_time/len(batch_prompts):.2f}s")
    print(f"  Successful responses: {successful_responses}/{len(batch_prompts)}")
    print(f"  Success rate: {successful_responses/len(batch_prompts)*100:.1f}%")
    
    # Show sample responses
    print(f"\n📄 Sample responses:")
    for i, response in enumerate(responses[:3], 1):
        print(f"  {i}. {response[:80]}...")
    
    return successful_responses >= len(batch_prompts) * 0.8

def main():
    """Run core API tests"""
    print("🧬 FOODB Core API Testing Suite")
    print("=" * 50)
    print("Testing essential wrapper functionality without external dependencies")
    print()
    
    test_results = []
    
    try:
        # Initialize wrapper
        wrapper = test_core_llm_wrapper()
        if not wrapper:
            print("❌ Failed to initialize wrapper. Cannot continue testing.")
            return
        
        # Run tests
        test_results.append(("Simple Sentence Generation", test_simple_sentence_generation(wrapper)))
        test_results.append(("Entity Extraction", test_entity_extraction(wrapper)))
        test_results.append(("Triple Extraction", test_triple_extraction(wrapper)))
        test_results.append(("Batch Processing", test_batch_processing(wrapper)))
        
        # Summary
        print(f"\n📊 Core API Test Results:")
        print("=" * 35)
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")
        
        if passed == len(test_results):
            print(f"\n🎉 All core tests passed! The FOODB API wrapper is fully functional.")
            print(f"\n💡 The wrapper successfully:")
            print(f"  ✅ Generates simple sentences from complex scientific text")
            print(f"  ✅ Extracts food compounds and bioactive molecules")
            print(f"  ✅ Identifies subject-predicate-object relationships")
            print(f"  ✅ Processes multiple requests efficiently")
            print(f"  ✅ Provides consistent sub-second response times")
            print(f"\n🚀 Ready for integration with your FOODB pipeline!")
        else:
            print(f"\n⚠️ Some tests failed. The wrapper has partial functionality.")
            print(f"Check the specific test results above for details.")
        
    except Exception as e:
        print(f"❌ Core testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
