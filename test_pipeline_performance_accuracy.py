#!/usr/bin/env python3
"""
FOODB Pipeline Performance and Accuracy Testing
This script tests the updated pipeline for time performance and accuracy
with the enhanced fallback system.
"""

import time
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import statistics

# Add pipeline directory to path
sys.path.append('FOODB_LLM_pipeline')

def create_test_data():
    """Create test data for pipeline testing"""
    print("📝 Creating test data...")
    
    test_sentences = [
        "Red wine contains resveratrol, a polyphenolic compound with antioxidant properties.",
        "Green tea is rich in catechins, particularly epigallocatechin gallate (EGCG).",
        "Blueberries contain anthocyanins, which give them their blue color and antioxidant activity.",
        "Turmeric contains curcumin, a bioactive compound with anti-inflammatory effects.",
        "Dark chocolate contains flavonoids, including epicatechin and procyanidins.",
        "Broccoli is a source of sulforaphane, a glucosinolate with potential health benefits.",
        "Tomatoes contain lycopene, a carotenoid pigment with antioxidant properties.",
        "Garlic contains allicin, an organosulfur compound formed when garlic is crushed.",
        "Citrus fruits are rich in vitamin C and flavonoids like hesperidin and naringin.",
        "Spinach contains lutein and zeaxanthin, carotenoids important for eye health."
    ]
    
    # Create test input file
    test_data = []
    for i, sentence in enumerate(test_sentences, 1):
        test_data.append({
            "id": f"test_{i}",
            "text": sentence,
            "source": "performance_test",
            "expected_compounds": []  # We'll fill this manually for accuracy testing
        })
    
    # Add expected compounds for accuracy testing
    expected_compounds = [
        ["resveratrol"],
        ["catechins", "epigallocatechin gallate", "EGCG"],
        ["anthocyanins"],
        ["curcumin"],
        ["flavonoids", "epicatechin", "procyanidins"],
        ["sulforaphane", "glucosinolate"],
        ["lycopene"],
        ["allicin"],
        ["vitamin C", "hesperidin", "naringin", "flavonoids"],
        ["lutein", "zeaxanthin"]
    ]
    
    for i, compounds in enumerate(expected_compounds):
        test_data[i]["expected_compounds"] = compounds
    
    # Save test data
    os.makedirs("FOODB_LLM_pipeline/sample_input", exist_ok=True)
    test_file = "FOODB_LLM_pipeline/sample_input/performance_test_data.jsonl"
    
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created test data: {test_file}")
    print(f"📊 Test sentences: {len(test_sentences)}")
    
    return test_file, test_data

def test_sentence_generation_performance():
    """Test the sentence generation script performance"""
    print("\n🧪 Testing Sentence Generation Performance")
    print("=" * 50)
    
    try:
        # Import the sentence generation script
        from llm_wrapper_enhanced import LLMWrapper
        
        # Initialize wrapper
        print("🔧 Initializing enhanced wrapper...")
        start_init = time.time()
        wrapper = LLMWrapper()
        end_init = time.time()
        
        print(f"✅ Wrapper initialized in {end_init - start_init:.2f}s")
        print(f"🎯 Primary provider: {wrapper.current_provider}")
        
        # Test data
        test_prompts = [
            "Extract metabolites from: Red wine contains resveratrol and anthocyanins.",
            "Extract metabolites from: Green tea contains catechins and EGCG.",
            "Extract metabolites from: Blueberries contain anthocyanins and vitamin C.",
            "Extract metabolites from: Turmeric contains curcumin and other compounds.",
            "Extract metabolites from: Dark chocolate contains flavonoids and epicatechin."
        ]
        
        print(f"\n🔬 Testing {len(test_prompts)} sentence generations...")
        
        results = []
        total_start = time.time()
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"  Request {i}: ", end="", flush=True)
            
            start_time = time.time()
            response = wrapper.generate_single(prompt, max_tokens=200)
            end_time = time.time()
            
            duration = end_time - start_time
            success = len(response) > 0
            
            results.append({
                'prompt': prompt,
                'response': response,
                'duration': duration,
                'success': success,
                'provider': wrapper.current_provider
            })
            
            print(f"{'✅' if success else '❌'} {duration:.2f}s [{wrapper.current_provider}]")
        
        total_end = time.time()
        total_duration = total_end - total_start
        
        # Calculate statistics
        durations = [r['duration'] for r in results if r['success']]
        successful = sum(1 for r in results if r['success'])
        
        print(f"\n📊 Sentence Generation Results:")
        print(f"  Total time: {total_duration:.2f}s")
        print(f"  Success rate: {successful}/{len(test_prompts)} ({successful/len(test_prompts):.1%})")
        print(f"  Average response time: {statistics.mean(durations):.2f}s")
        print(f"  Min response time: {min(durations):.2f}s")
        print(f"  Max response time: {max(durations):.2f}s")
        print(f"  Throughput: {len(test_prompts)/total_duration:.1f} requests/second")
        
        # Check for provider switches
        stats = wrapper.get_statistics()
        if stats['fallback_switches'] > 0:
            print(f"  🔄 Provider switches: {stats['fallback_switches']}")
            print(f"  🚨 Rate limited requests: {stats['rate_limited_requests']}")
        
        return {
            'total_duration': total_duration,
            'success_rate': successful/len(test_prompts),
            'avg_response_time': statistics.mean(durations),
            'throughput': len(test_prompts)/total_duration,
            'results': results,
            'stats': stats
        }
        
    except Exception as e:
        print(f"❌ Error testing sentence generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_accuracy_with_expected_results():
    """Test accuracy by comparing with expected compound extractions"""
    print("\n🎯 Testing Extraction Accuracy")
    print("=" * 35)
    
    try:
        from llm_wrapper_enhanced import LLMWrapper
        
        wrapper = LLMWrapper()
        
        # Test cases with expected results
        test_cases = [
            {
                'text': "Red wine contains resveratrol and anthocyanins.",
                'expected': ["resveratrol", "anthocyanins"],
                'description': "Simple compound extraction"
            },
            {
                'text': "Green tea is rich in catechins, particularly epigallocatechin gallate (EGCG).",
                'expected': ["catechins", "epigallocatechin gallate", "EGCG"],
                'description': "Multiple compounds with abbreviation"
            },
            {
                'text': "Turmeric contains curcumin, a bioactive compound with anti-inflammatory effects.",
                'expected': ["curcumin"],
                'description': "Single compound with description"
            },
            {
                'text': "Dark chocolate contains flavonoids, including epicatechin and procyanidins.",
                'expected': ["flavonoids", "epicatechin", "procyanidins"],
                'description': "Compound class and specific compounds"
            },
            {
                'text': "Citrus fruits are rich in vitamin C and flavonoids like hesperidin and naringin.",
                'expected': ["vitamin C", "flavonoids", "hesperidin", "naringin"],
                'description': "Vitamins and specific flavonoids"
            }
        ]
        
        print(f"🔬 Testing {len(test_cases)} accuracy cases...")
        
        accuracy_results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n  Test {i}: {case['description']}")
            print(f"    Input: {case['text']}")
            
            prompt = f"Extract metabolites from: {case['text']}"
            
            start_time = time.time()
            response = wrapper.generate_single(prompt, max_tokens=150)
            end_time = time.time()
            
            print(f"    Response: {response}")
            print(f"    Expected: {case['expected']}")
            
            # Simple accuracy check - count how many expected compounds are mentioned
            response_lower = response.lower()
            found_compounds = []
            
            for compound in case['expected']:
                if compound.lower() in response_lower:
                    found_compounds.append(compound)
            
            accuracy = len(found_compounds) / len(case['expected'])
            
            accuracy_results.append({
                'case': case['description'],
                'expected_count': len(case['expected']),
                'found_count': len(found_compounds),
                'found_compounds': found_compounds,
                'accuracy': accuracy,
                'response_time': end_time - start_time,
                'response': response
            })
            
            print(f"    Found: {found_compounds}")
            print(f"    Accuracy: {accuracy:.1%} ({len(found_compounds)}/{len(case['expected'])})")
            print(f"    Time: {end_time - start_time:.2f}s")
        
        # Calculate overall accuracy
        total_expected = sum(r['expected_count'] for r in accuracy_results)
        total_found = sum(r['found_count'] for r in accuracy_results)
        overall_accuracy = total_found / total_expected
        
        print(f"\n📊 Accuracy Results:")
        print(f"  Overall accuracy: {overall_accuracy:.1%} ({total_found}/{total_expected})")
        print(f"  Average per-case accuracy: {statistics.mean([r['accuracy'] for r in accuracy_results]):.1%}")
        print(f"  Perfect extractions: {sum(1 for r in accuracy_results if r['accuracy'] == 1.0)}/{len(accuracy_results)}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'avg_case_accuracy': statistics.mean([r['accuracy'] for r in accuracy_results]),
            'perfect_cases': sum(1 for r in accuracy_results if r['accuracy'] == 1.0),
            'results': accuracy_results
        }
        
    except Exception as e:
        print(f"❌ Error testing accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_stress_performance():
    """Test performance under stress (rapid requests)"""
    print("\n🔥 Testing Stress Performance")
    print("=" * 30)
    
    try:
        from llm_wrapper_enhanced import LLMWrapper
        
        wrapper = LLMWrapper()
        
        # Create 20 rapid requests
        stress_prompts = [
            f"Extract metabolites from sample {i}: Contains various bioactive compounds."
            for i in range(1, 21)
        ]
        
        print(f"🚀 Sending {len(stress_prompts)} rapid requests...")
        
        stress_results = []
        start_stress = time.time()
        
        for i, prompt in enumerate(stress_prompts, 1):
            print(f"  Request {i:2d}: ", end="", flush=True)
            
            provider_before = wrapper.current_provider
            
            start_time = time.time()
            response = wrapper.generate_single(prompt, max_tokens=100)
            end_time = time.time()
            
            provider_after = wrapper.current_provider
            success = len(response) > 0
            
            stress_results.append({
                'request': i,
                'provider_before': provider_before,
                'provider_after': provider_after,
                'success': success,
                'duration': end_time - start_time,
                'response_length': len(response)
            })
            
            print(f"{'✅' if success else '❌'} {end_time - start_time:.2f}s [{provider_after}]")
            
            if provider_before != provider_after:
                print(f"    🔄 Provider switch: {provider_before} → {provider_after}")
        
        end_stress = time.time()
        total_stress_time = end_stress - start_stress
        
        # Analyze stress results
        successful_stress = sum(1 for r in stress_results if r['success'])
        avg_stress_time = statistics.mean([r['duration'] for r in stress_results if r['success']])
        
        # Count provider switches
        switches = sum(1 for r in stress_results if r['provider_before'] != r['provider_after'])
        
        print(f"\n📊 Stress Test Results:")
        print(f"  Total time: {total_stress_time:.2f}s")
        print(f"  Success rate: {successful_stress}/{len(stress_prompts)} ({successful_stress/len(stress_prompts):.1%})")
        print(f"  Average response time: {avg_stress_time:.2f}s")
        print(f"  Throughput: {len(stress_prompts)/total_stress_time:.1f} requests/second")
        print(f"  Provider switches: {switches}")
        
        # Final statistics
        final_stats = wrapper.get_statistics()
        print(f"  Rate limited requests: {final_stats['rate_limited_requests']}")
        print(f"  Total fallback switches: {final_stats['fallback_switches']}")
        
        return {
            'total_time': total_stress_time,
            'success_rate': successful_stress/len(stress_prompts),
            'avg_response_time': avg_stress_time,
            'throughput': len(stress_prompts)/total_stress_time,
            'provider_switches': switches,
            'final_stats': final_stats
        }
        
    except Exception as e:
        print(f"❌ Error in stress test: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_performance_report(sentence_results, accuracy_results, stress_results):
    """Save comprehensive performance report"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = f"FOODB_Pipeline_Performance_Report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'test_type': 'foodb_pipeline_performance_accuracy',
        'sentence_generation': sentence_results,
        'accuracy_testing': accuracy_results,
        'stress_testing': stress_results,
        'summary': {
            'overall_performance': 'excellent' if (
                sentence_results and sentence_results['success_rate'] > 0.9 and
                accuracy_results and accuracy_results['overall_accuracy'] > 0.8 and
                stress_results and stress_results['success_rate'] > 0.9
            ) else 'needs_improvement'
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Performance report saved: {report_file}")
    return report_file

def main():
    """Run comprehensive performance and accuracy testing"""
    print("🧪 FOODB Pipeline - Performance and Accuracy Testing")
    print("=" * 60)
    
    try:
        # Test 1: Sentence generation performance
        sentence_results = test_sentence_generation_performance()
        
        # Test 2: Accuracy testing
        accuracy_results = test_accuracy_with_expected_results()
        
        # Test 3: Stress performance
        stress_results = test_stress_performance()
        
        # Save comprehensive report
        if sentence_results and accuracy_results and stress_results:
            report_file = save_performance_report(sentence_results, accuracy_results, stress_results)
        
        # Final summary
        print(f"\n🎉 PERFORMANCE AND ACCURACY TESTING COMPLETE!")
        print("=" * 55)
        
        if sentence_results:
            print(f"📊 Sentence Generation:")
            print(f"  Success rate: {sentence_results['success_rate']:.1%}")
            print(f"  Avg response time: {sentence_results['avg_response_time']:.2f}s")
            print(f"  Throughput: {sentence_results['throughput']:.1f} req/s")
        
        if accuracy_results:
            print(f"🎯 Accuracy:")
            print(f"  Overall accuracy: {accuracy_results['overall_accuracy']:.1%}")
            print(f"  Perfect extractions: {accuracy_results['perfect_cases']}/5")
        
        if stress_results:
            print(f"🔥 Stress Performance:")
            print(f"  Success rate: {stress_results['success_rate']:.1%}")
            print(f"  Provider switches: {stress_results['provider_switches']}")
            print(f"  Throughput: {stress_results['throughput']:.1f} req/s")
        
        # Overall assessment
        if (sentence_results and sentence_results['success_rate'] > 0.9 and
            accuracy_results and accuracy_results['overall_accuracy'] > 0.8 and
            stress_results and stress_results['success_rate'] > 0.9):
            print(f"\n✅ OVERALL ASSESSMENT: EXCELLENT PERFORMANCE")
            print(f"  The enhanced FOODB pipeline demonstrates:")
            print(f"  • High reliability (>90% success rate)")
            print(f"  • Good accuracy (>80% compound extraction)")
            print(f"  • Stress resilience (handles rapid requests)")
            print(f"  • Effective fallback system")
        else:
            print(f"\n⚠️ OVERALL ASSESSMENT: NEEDS IMPROVEMENT")
            print(f"  Some performance metrics below target thresholds")
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
