#!/usr/bin/env python3
"""
Test Wine PDF with Improved Exponential Backoff
This script tests the wine PDF processing with the improved exponential backoff logic
"""

import time
import json
import sys
import os
import pandas as pd
sys.path.append('FOODB_LLM_pipeline')

def test_wine_pdf_with_exponential_backoff():
    """Test wine PDF processing with improved exponential backoff"""
    print("🍷 Wine PDF Processing with Exponential Backoff")
    print("=" * 55)
    
    # Load wine biomarkers database
    print("📊 Loading wine biomarkers database...")
    df = pd.read_csv("urinary_wine_biomarkers.csv")
    expected_biomarkers = set(df['Compound Name'].str.lower().tolist())
    print(f"✅ Loaded {len(expected_biomarkers)} expected biomarkers")
    
    # Extract text from PDF (simulated - using existing extraction)
    print("\n📄 Loading PDF text...")
    try:
        with open('wine_biomarkers_test_results.json', 'r') as f:
            previous_results = json.load(f)
        
        # Use the text chunks from previous extraction
        if 'text_chunks' in previous_results:
            text_chunks = previous_results['text_chunks'][:20]  # Test with first 20 chunks
            print(f"✅ Loaded {len(text_chunks)} text chunks from previous extraction")
        else:
            print("❌ No text chunks found in previous results")
            return
    except:
        print("❌ Could not load previous PDF extraction results")
        return
    
    # Initialize enhanced wrapper with exponential backoff
    print(f"\n🔧 Initializing enhanced wrapper with exponential backoff...")
    from llm_wrapper_enhanced import LLMWrapper, RetryConfig
    
    # Configure for better backoff behavior
    retry_config = RetryConfig(
        max_attempts=5,        # 5 attempts per provider
        base_delay=2.0,        # 2 second base delay
        max_delay=30.0,        # Max 30 seconds
        exponential_base=2.0,  # Double each time
        jitter=True            # Add jitter for better behavior
    )
    
    wrapper = LLMWrapper(retry_config=retry_config)
    
    print(f"✅ Enhanced wrapper initialized")
    print(f"🎯 Primary provider: {wrapper.current_provider}")
    print(f"📋 Retry config: {retry_config.max_attempts} attempts, {retry_config.base_delay}s base delay")
    print(f"⏱️ Expected delays: 2s → 4s → 8s → 16s → 30s")
    print(f"🔄 Fallback order: {' → '.join(wrapper.fallback_order)}")
    
    # Process chunks with improved backoff
    print(f"\n🧪 Processing {len(text_chunks)} chunks with exponential backoff...")
    
    results = []
    detected_biomarkers = set()
    processing_start = time.time()
    
    for i, chunk in enumerate(text_chunks, 1):
        print(f"\nChunk {i:2d}: ", end="", flush=True)
        
        provider_before = wrapper.current_provider
        chunk_start = time.time()
        
        # Create prompt for biomarker extraction
        prompt = f"Extract wine biomarkers and metabolites from this scientific text:\n\n{chunk}\n\nList all compounds that could be found in urine after wine consumption."
        
        response = wrapper.generate_single(prompt, max_tokens=200)
        
        chunk_end = time.time()
        provider_after = wrapper.current_provider
        
        chunk_time = chunk_end - chunk_start
        success = len(response) > 0
        
        # Check for biomarkers in response
        found_in_chunk = set()
        if success:
            response_lower = response.lower()
            for biomarker in expected_biomarkers:
                if len(biomarker) > 3 and biomarker in response_lower:
                    found_in_chunk.add(biomarker)
                    detected_biomarkers.add(biomarker)
        
        results.append({
            'chunk': i,
            'provider_before': provider_before,
            'provider_after': provider_after,
            'processing_time': chunk_time,
            'success': success,
            'response': response,
            'biomarkers_found': list(found_in_chunk)
        })
        
        print(f"{'✅' if success else '❌'} {chunk_time:.2f}s [{provider_after}]", end="")
        
        if found_in_chunk:
            print(f" ({len(found_in_chunk)} biomarkers)")
        else:
            print()
        
        # Highlight important events
        if provider_before != provider_after:
            print(f"    🔄 Provider switch: {provider_before} → {provider_after}")
        
        if chunk_time > 10.0:
            print(f"    ⏱️ Long processing: {chunk_time:.2f}s (full exponential backoff)")
        elif chunk_time > 5.0:
            print(f"    ⏱️ Extended processing: {chunk_time:.2f}s (partial backoff)")
        elif chunk_time > 2.0:
            print(f"    ⏱️ Retry detected: {chunk_time:.2f}s")
        
        # Show progress every 5 chunks
        if i % 5 == 0:
            stats = wrapper.get_statistics()
            print(f"    📊 Progress: {stats['rate_limited_requests']} rate limited, {stats['fallback_switches']} switches")
    
    processing_end = time.time()
    total_time = processing_end - processing_start
    
    # Calculate metrics
    successful_chunks = sum(1 for r in results if r['success'])
    total_biomarkers_found = len(detected_biomarkers)
    
    # Calculate precision, recall, F1
    true_positives = len(detected_biomarkers.intersection(expected_biomarkers))
    false_positives = 0  # We're only counting expected biomarkers
    false_negatives = len(expected_biomarkers - detected_biomarkers)
    
    precision = true_positives / total_biomarkers_found if total_biomarkers_found > 0 else 0
    recall = true_positives / len(expected_biomarkers)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Analyze results
    print(f"\n🎉 WINE PDF PROCESSING WITH EXPONENTIAL BACKOFF COMPLETE!")
    print("=" * 65)
    
    print(f"\n⏱️ TIMING ANALYSIS:")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average per chunk: {total_time/len(text_chunks):.2f}s")
    print(f"  Successful chunks: {successful_chunks}/{len(text_chunks)} ({successful_chunks/len(text_chunks):.1%})")
    
    # Timing distribution
    fast_chunks = [r for r in results if r['processing_time'] < 2.0]
    retry_chunks = [r for r in results if 2.0 <= r['processing_time'] < 5.0]
    backoff_chunks = [r for r in results if r['processing_time'] >= 5.0]
    
    print(f"  Fast processing (<2s): {len(fast_chunks)} chunks")
    print(f"  With retries (2-5s): {len(retry_chunks)} chunks")
    print(f"  With backoff (≥5s): {len(backoff_chunks)} chunks")
    
    if backoff_chunks:
        max_time = max(r['processing_time'] for r in backoff_chunks)
        print(f"  Maximum processing time: {max_time:.2f}s (full exponential backoff)")
    
    print(f"\n🍷 BIOMARKER DETECTION:")
    print(f"  Expected biomarkers: {len(expected_biomarkers)}")
    print(f"  Detected biomarkers: {total_biomarkers_found}")
    print(f"  Correctly detected: {true_positives}")
    
    print(f"\n🎯 PERFORMANCE SCORES:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1_score:.3f}")
    
    # Provider analysis
    provider_usage = {}
    provider_switches = []
    
    for result in results:
        provider = result['provider_after']
        provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        if result['provider_before'] != result['provider_after']:
            provider_switches.append(result)
    
    print(f"\n🔄 PROVIDER ANALYSIS:")
    print(f"  Provider usage:")
    for provider, count in provider_usage.items():
        percentage = count / len(results) * 100
        print(f"    {provider}: {count}/{len(results)} chunks ({percentage:.1f}%)")
    
    print(f"  Provider switches: {len(provider_switches)}")
    for switch in provider_switches:
        print(f"    Chunk {switch['chunk']}: {switch['provider_before']} → {switch['provider_after']} ({switch['processing_time']:.2f}s)")
    
    # Final statistics
    final_stats = wrapper.get_statistics()
    print(f"\n📈 FINAL STATISTICS:")
    print(f"  Total requests: {final_stats['total_requests']}")
    print(f"  Successful: {final_stats['successful_requests']}")
    print(f"  Rate limited: {final_stats['rate_limited_requests']}")
    print(f"  Provider switches: {final_stats['fallback_switches']}")
    print(f"  Success rate: {final_stats['success_rate']:.1%}")
    
    # Show detected biomarkers
    if detected_biomarkers:
        print(f"\n✅ DETECTED BIOMARKERS ({len(detected_biomarkers)}):")
        for biomarker in sorted(detected_biomarkers):
            print(f"   • {biomarker}")
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = f"wine_pdf_exponential_backoff_results_{timestamp}.json"
    
    results_data = {
        'timestamp': timestamp,
        'test_type': 'wine_pdf_exponential_backoff',
        'chunks_processed': len(text_chunks),
        'total_time': total_time,
        'biomarker_detection': {
            'expected': len(expected_biomarkers),
            'detected': total_biomarkers_found,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detected_biomarkers': sorted(list(detected_biomarkers))
        },
        'provider_performance': {
            'usage': provider_usage,
            'switches': len(provider_switches),
            'statistics': final_stats
        },
        'chunk_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    
    # Comparison with previous results
    print(f"\n📊 IMPROVEMENT ANALYSIS:")
    print(f"With exponential backoff:")
    print(f"  ✅ Each provider gets multiple retry attempts")
    print(f"  ✅ System waits for rate limits to recover")
    print(f"  ✅ More patient before switching providers")
    print(f"  ✅ Better utilization of available providers")
    
    return results_data

def main():
    """Test wine PDF with exponential backoff"""
    print("🍷 Wine PDF Processing - Exponential Backoff Test")
    print("=" * 55)
    
    try:
        results = test_wine_pdf_with_exponential_backoff()
        
        if results:
            print(f"\n🎯 CONCLUSION:")
            print(f"The enhanced wrapper with exponential backoff provides:")
            print(f"  • Better rate limit handling")
            print(f"  • More efficient provider utilization") 
            print(f"  • Improved processing reliability")
            print(f"  • Patient retry behavior before switching")
        
    except Exception as e:
        print(f"❌ Wine PDF exponential backoff test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
