#!/usr/bin/env python3
"""
Test Triple Extraction Script Performance
This script specifically tests the simple_sentenceRE3_API.py performance and accuracy
"""

import time
import json
import sys
import os
from pathlib import Path

# Add pipeline directory to path
sys.path.append('FOODB_LLM_pipeline')

def create_triple_test_data():
    """Create test data for triple extraction"""
    print("📝 Creating triple extraction test data...")
    
    # Create test sentences with expected triples
    test_sentences = [
        {
            "text": "Resveratrol is found in red wine.",
            "expected_triples": [("resveratrol", "found_in", "red wine")]
        },
        {
            "text": "Curcumin exhibits anti-inflammatory properties.",
            "expected_triples": [("curcumin", "exhibits", "anti-inflammatory properties")]
        },
        {
            "text": "Green tea contains catechins.",
            "expected_triples": [("green tea", "contains", "catechins")]
        },
        {
            "text": "Anthocyanins give blueberries their blue color.",
            "expected_triples": [("anthocyanins", "give", "blueberries their blue color")]
        },
        {
            "text": "Lycopene is a carotenoid found in tomatoes.",
            "expected_triples": [("lycopene", "is", "carotenoid"), ("lycopene", "found_in", "tomatoes")]
        }
    ]
    
    # Create input file for triple extraction
    os.makedirs("FOODB_LLM_pipeline/sample_input", exist_ok=True)
    test_file = "FOODB_LLM_pipeline/sample_input/triple_test_data.jsonl"
    
    with open(test_file, 'w') as f:
        for i, item in enumerate(test_sentences, 1):
            test_item = {
                "id": f"triple_test_{i}",
                "text": item["text"],
                "source": "triple_performance_test"
            }
            f.write(json.dumps(test_item) + '\n')
    
    print(f"✅ Created triple test data: {test_file}")
    return test_file, test_sentences

def test_triple_extraction_wrapper():
    """Test the triple extraction wrapper directly"""
    print("\n🧪 Testing Triple Extraction Wrapper")
    print("=" * 40)
    
    try:
        # Import and test the wrapper loading function
        from simple_sentenceRE3_API import load_api_wrapper
        
        print("🔧 Loading API wrapper...")
        start_time = time.time()
        
        # This should show the enhanced wrapper status
        wrapper = load_api_wrapper()
        
        end_time = time.time()
        
        if wrapper:
            print(f"✅ Wrapper loaded successfully in {end_time - start_time:.2f}s")
            
            # Test basic functionality
            test_prompt = "Extract relationship: Resveratrol is found in red wine."
            
            print(f"🔬 Testing basic generation...")
            start_gen = time.time()
            response = wrapper.generate_single(test_prompt, max_tokens=100)
            end_gen = time.time()
            
            if response:
                print(f"✅ Generation successful in {end_gen - start_gen:.2f}s")
                print(f"📝 Response: {response[:100]}...")
                
                # Check statistics
                if hasattr(wrapper, 'get_statistics'):
                    stats = wrapper.get_statistics()
                    print(f"📊 Statistics: {stats['total_requests']} requests, {stats['success_rate']:.1%} success")
                
                return wrapper
            else:
                print(f"❌ Generation failed")
                return None
        else:
            print(f"❌ Wrapper loading failed")
            return None
            
    except Exception as e:
        print(f"❌ Error testing triple extraction wrapper: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_triple_extraction_performance():
    """Test performance of triple extraction with multiple requests"""
    print("\n🔬 Testing Triple Extraction Performance")
    print("=" * 45)
    
    try:
        from simple_sentenceRE3_API import load_api_wrapper
        
        wrapper = load_api_wrapper()
        if not wrapper:
            print("❌ Could not load wrapper")
            return None
        
        # Test prompts for triple extraction
        test_prompts = [
            "Extract relationships: Resveratrol is found in red wine and has antioxidant properties.",
            "Extract relationships: Curcumin is present in turmeric and exhibits anti-inflammatory effects.",
            "Extract relationships: Green tea contains catechins which provide health benefits.",
            "Extract relationships: Anthocyanins are found in blueberries and give them blue color.",
            "Extract relationships: Lycopene is a carotenoid present in tomatoes.",
            "Extract relationships: Quercetin is a flavonoid found in onions and apples.",
            "Extract relationships: Sulforaphane is present in broccoli and has protective effects.",
            "Extract relationships: Allicin is found in garlic and has antimicrobial properties."
        ]
        
        print(f"🚀 Testing {len(test_prompts)} triple extractions...")
        
        results = []
        total_start = time.time()
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"  Request {i}: ", end="", flush=True)
            
            provider_before = getattr(wrapper, 'current_provider', 'unknown')
            
            start_time = time.time()
            response = wrapper.generate_single(prompt, max_tokens=200)
            end_time = time.time()
            
            provider_after = getattr(wrapper, 'current_provider', 'unknown')
            duration = end_time - start_time
            success = len(response) > 0
            
            results.append({
                'prompt': prompt,
                'response': response,
                'duration': duration,
                'success': success,
                'provider_before': provider_before,
                'provider_after': provider_after
            })
            
            print(f"{'✅' if success else '❌'} {duration:.2f}s [{provider_after}]")
            
            if provider_before != provider_after:
                print(f"    🔄 Provider switch: {provider_before} → {provider_after}")
        
        total_end = time.time()
        total_duration = total_end - total_start
        
        # Calculate statistics
        successful = sum(1 for r in results if r['success'])
        durations = [r['duration'] for r in results if r['success']]
        
        print(f"\n📊 Triple Extraction Performance:")
        print(f"  Total time: {total_duration:.2f}s")
        print(f"  Success rate: {successful}/{len(test_prompts)} ({successful/len(test_prompts):.1%})")
        
        if durations:
            import statistics
            print(f"  Average response time: {statistics.mean(durations):.2f}s")
            print(f"  Min response time: {min(durations):.2f}s")
            print(f"  Max response time: {max(durations):.2f}s")
            print(f"  Throughput: {len(test_prompts)/total_duration:.1f} requests/second")
        
        # Check for provider switches
        switches = sum(1 for r in results if r['provider_before'] != r['provider_after'])
        if switches > 0:
            print(f"  🔄 Provider switches: {switches}")
        
        # Check wrapper statistics if available
        if hasattr(wrapper, 'get_statistics'):
            stats = wrapper.get_statistics()
            print(f"  📈 Wrapper stats: {stats['total_requests']} total, {stats['fallback_switches']} switches")
        
        return {
            'total_duration': total_duration,
            'success_rate': successful/len(test_prompts),
            'avg_response_time': statistics.mean(durations) if durations else 0,
            'throughput': len(test_prompts)/total_duration,
            'provider_switches': switches,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Error testing triple extraction performance: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_triple_accuracy():
    """Test accuracy of triple extraction"""
    print("\n🎯 Testing Triple Extraction Accuracy")
    print("=" * 40)
    
    try:
        from simple_sentenceRE3_API import load_api_wrapper
        
        wrapper = load_api_wrapper()
        if not wrapper:
            print("❌ Could not load wrapper")
            return None
        
        # Test cases with expected relationships
        test_cases = [
            {
                'text': "Resveratrol is found in red wine.",
                'expected_entities': ["resveratrol", "red wine"],
                'expected_relation': "found_in",
                'description': "Simple compound-source relationship"
            },
            {
                'text': "Curcumin exhibits anti-inflammatory properties.",
                'expected_entities': ["curcumin", "anti-inflammatory properties"],
                'expected_relation': "exhibits",
                'description': "Compound-property relationship"
            },
            {
                'text': "Green tea contains catechins.",
                'expected_entities': ["green tea", "catechins"],
                'expected_relation': "contains",
                'description': "Source-compound relationship"
            }
        ]
        
        print(f"🔬 Testing {len(test_cases)} accuracy cases...")
        
        accuracy_results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n  Test {i}: {case['description']}")
            print(f"    Input: {case['text']}")
            
            prompt = f"Extract relationships from: {case['text']}"
            
            start_time = time.time()
            response = wrapper.generate_single(prompt, max_tokens=150)
            end_time = time.time()
            
            print(f"    Response: {response}")
            print(f"    Expected entities: {case['expected_entities']}")
            print(f"    Expected relation: {case['expected_relation']}")
            
            # Simple accuracy check - look for expected entities and relation
            response_lower = response.lower()
            
            found_entities = []
            for entity in case['expected_entities']:
                if entity.lower() in response_lower:
                    found_entities.append(entity)
            
            relation_found = case['expected_relation'].lower() in response_lower
            
            entity_accuracy = len(found_entities) / len(case['expected_entities'])
            overall_accuracy = (entity_accuracy + (1 if relation_found else 0)) / 2
            
            accuracy_results.append({
                'case': case['description'],
                'entity_accuracy': entity_accuracy,
                'relation_found': relation_found,
                'overall_accuracy': overall_accuracy,
                'response_time': end_time - start_time,
                'found_entities': found_entities
            })
            
            print(f"    Found entities: {found_entities} ({entity_accuracy:.1%})")
            print(f"    Relation found: {'✅' if relation_found else '❌'}")
            print(f"    Overall accuracy: {overall_accuracy:.1%}")
            print(f"    Time: {end_time - start_time:.2f}s")
        
        # Calculate overall accuracy
        if accuracy_results:
            import statistics
            avg_accuracy = statistics.mean([r['overall_accuracy'] for r in accuracy_results])
            perfect_cases = sum(1 for r in accuracy_results if r['overall_accuracy'] == 1.0)
            
            print(f"\n📊 Triple Extraction Accuracy:")
            print(f"  Average accuracy: {avg_accuracy:.1%}")
            print(f"  Perfect extractions: {perfect_cases}/{len(test_cases)}")
            
            return {
                'avg_accuracy': avg_accuracy,
                'perfect_cases': perfect_cases,
                'total_cases': len(test_cases),
                'results': accuracy_results
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error testing triple accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_triple_performance_report(wrapper_test, performance_test, accuracy_test):
    """Save triple extraction performance report"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = f"Triple_Extraction_Performance_Report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'test_type': 'triple_extraction_performance',
        'wrapper_test': wrapper_test is not None,
        'performance_test': performance_test,
        'accuracy_test': accuracy_test,
        'summary': {
            'performance_rating': 'excellent' if (
                performance_test and performance_test['success_rate'] > 0.9 and
                accuracy_test and accuracy_test['avg_accuracy'] > 0.7
            ) else 'good' if (
                performance_test and performance_test['success_rate'] > 0.8
            ) else 'needs_improvement'
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Triple extraction report saved: {report_file}")
    return report_file

def main():
    """Run comprehensive triple extraction testing"""
    print("🧪 FOODB Pipeline - Triple Extraction Performance Testing")
    print("=" * 65)
    
    try:
        # Test 1: Wrapper functionality
        wrapper_test = test_triple_extraction_wrapper()
        
        # Test 2: Performance testing
        performance_test = test_triple_extraction_performance()
        
        # Test 3: Accuracy testing
        accuracy_test = test_triple_accuracy()
        
        # Save report
        if performance_test or accuracy_test:
            report_file = save_triple_performance_report(wrapper_test, performance_test, accuracy_test)
        
        # Final summary
        print(f"\n🎉 TRIPLE EXTRACTION TESTING COMPLETE!")
        print("=" * 45)
        
        if performance_test:
            print(f"📊 Performance:")
            print(f"  Success rate: {performance_test['success_rate']:.1%}")
            print(f"  Avg response time: {performance_test['avg_response_time']:.2f}s")
            print(f"  Throughput: {performance_test['throughput']:.1f} req/s")
            print(f"  Provider switches: {performance_test['provider_switches']}")
        
        if accuracy_test:
            print(f"🎯 Accuracy:")
            print(f"  Average accuracy: {accuracy_test['avg_accuracy']:.1%}")
            print(f"  Perfect extractions: {accuracy_test['perfect_cases']}/{accuracy_test['total_cases']}")
        
        # Overall assessment
        if (performance_test and performance_test['success_rate'] > 0.9 and
            accuracy_test and accuracy_test['avg_accuracy'] > 0.7):
            print(f"\n✅ OVERALL ASSESSMENT: EXCELLENT")
            print(f"  Triple extraction system is performing well")
        elif performance_test and performance_test['success_rate'] > 0.8:
            print(f"\n✅ OVERALL ASSESSMENT: GOOD")
            print(f"  Triple extraction system is reliable")
        else:
            print(f"\n⚠️ OVERALL ASSESSMENT: NEEDS IMPROVEMENT")
        
    except Exception as e:
        print(f"❌ Triple extraction testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
