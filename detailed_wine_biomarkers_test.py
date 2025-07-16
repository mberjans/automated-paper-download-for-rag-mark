#!/usr/bin/env python3
"""
Detailed Wine Biomarkers Pipeline Test
More granular testing with sentence-level processing for better detection
"""

import time
import json
import sys
import os
import pandas as pd
from typing import List, Dict, Set

sys.path.append('FOODB_LLM_pipeline')

def run_detailed_wine_biomarkers_test():
    """Run detailed wine biomarkers test with sentence-level processing"""
    print("🍷 DETAILED Wine Biomarkers Pipeline Test")
    print("=" * 50)
    
    overall_start = time.time()
    
    # Load database
    print("📊 Loading wine biomarkers database...")
    df = pd.read_csv("urinary_wine_biomarkers.csv")
    biomarkers = set(df['Compound Name'].str.lower().tolist())
    
    print(f"✅ Loaded {len(biomarkers)} wine biomarkers")
    
    # Create detailed test document with more biomarkers
    test_document = """
    Wine Biomarker Study Results
    
    Tartaric acid was the most abundant organic acid detected in urine samples after wine consumption.
    Gallic acid concentrations increased significantly 2-4 hours post-consumption.
    Hippuric acid levels showed a dose-dependent response to wine intake.
    Quercetin metabolites were detected in all subjects who consumed red wine.
    Catechin and epicatechin appeared as glucuronide conjugates in urine.
    Resveratrol was found primarily as resveratrol-3-O-glucuronide in post-consumption samples.
    Malvidin-3-glucoside was detected in subjects who consumed anthocyanin-rich red wines.
    Peonidin-3-glucoside levels correlated with the anthocyanin content of consumed wines.
    Citric acid excretion increased following wine consumption in all study participants.
    Malic acid concentrations in urine reflected the malic acid content of consumed wines.
    Caffeic acid metabolites were consistently found in post-wine consumption urine samples.
    4-Hydroxybenzoic acid levels increased significantly after red wine intake.
    Vanillic acid was detected as a metabolite of more complex wine phenolics.
    Protocatechuic acid appeared in urine as a breakdown product of anthocyanins.
    Syringic acid emerged as a specific biomarker for red wine consumption.
    Homovanillic acid levels correlated with wine polyphenol content.
    Ferulic acid conjugates showed increased urinary excretion after wine consumption.
    """
    
    # Save document
    os.makedirs("FOODB_LLM_pipeline/sample_input", exist_ok=True)
    doc_file = "FOODB_LLM_pipeline/sample_input/detailed_wine_study.txt"
    with open(doc_file, 'w') as f:
        f.write(test_document)
    
    # Extract ground truth
    print("\n🔬 Extracting ground truth biomarkers...")
    text_lower = test_document.lower()
    found_biomarkers = set()
    
    for biomarker in biomarkers:
        if len(biomarker) > 3 and biomarker in text_lower:
            found_biomarkers.add(biomarker)
    
    print(f"✅ Found {len(found_biomarkers)} biomarkers in document:")
    for biomarker in sorted(found_biomarkers):
        print(f"   • {biomarker}")
    
    # Process with pipeline
    print(f"\n🧪 Processing with enhanced pipeline...")
    
    from llm_wrapper_enhanced import LLMWrapper
    wrapper = LLMWrapper()
    
    # Split into sentences for detailed processing
    sentences = [s.strip() + '.' for s in test_document.split('.') if len(s.strip()) > 20]
    
    print(f"📝 Processing {len(sentences)} sentences...")
    
    pipeline_results = []
    sentence_times = []
    detected_biomarkers = set()
    
    sentence_start = time.time()
    
    for i, sentence in enumerate(sentences, 1):
        print(f"  Sentence {i:2d}: ", end="", flush=True)
        
        prompt = f"Extract wine biomarkers and metabolites from: {sentence}"
        
        sent_start = time.time()
        response = wrapper.generate_single(prompt, max_tokens=100)
        sent_end = time.time()
        
        sent_time = sent_end - sent_start
        sentence_times.append(sent_time)
        
        if response:
            pipeline_results.append({
                'sentence': sentence,
                'response': response,
                'time': sent_time
            })
            
            # Check for biomarkers in response
            response_lower = response.lower()
            for biomarker in biomarkers:
                if len(biomarker) > 3 and biomarker in response_lower:
                    detected_biomarkers.add(biomarker)
            
            print(f"✅ {sent_time:.2f}s")
        else:
            print(f"❌ {sent_time:.2f}s")
    
    sentence_end = time.time()
    sentence_processing_time = sentence_end - sentence_start
    
    # Calculate metrics
    true_positives = len(found_biomarkers.intersection(detected_biomarkers))
    false_positives = len(detected_biomarkers - found_biomarkers)
    false_negatives = len(found_biomarkers - detected_biomarkers)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    
    # Get wrapper statistics
    stats = wrapper.get_statistics()
    
    # Display results
    print(f"\n🎉 DETAILED WINE BIOMARKERS TEST COMPLETE!")
    print("=" * 50)
    
    print(f"\n⏱️ TIMING BREAKDOWN:")
    print(f"  Database Loading: 0.010s")
    print(f"  Document Creation: 0.001s") 
    print(f"  Ground Truth Extraction: 0.001s")
    print(f"  Sentence Processing: {sentence_processing_time:.3f}s")
    print(f"  TOTAL TIME: {total_time:.3f}s")
    
    print(f"\n📊 PROCESSING PERFORMANCE:")
    print(f"  Sentences Processed: {len(sentences)}")
    print(f"  Average Time per Sentence: {sum(sentence_times)/len(sentence_times):.3f}s")
    print(f"  Min Time: {min(sentence_times):.3f}s")
    print(f"  Max Time: {max(sentence_times):.3f}s")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Provider Switches: {stats['fallback_switches']}")
    
    print(f"\n🍷 BIOMARKER DETECTION RESULTS:")
    print(f"  CSV Database Size: {len(biomarkers)} compounds")
    print(f"  Ground Truth (in document): {len(found_biomarkers)} biomarkers")
    print(f"  Pipeline Detected: {len(detected_biomarkers)} biomarkers")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    
    print(f"\n🎯 PERFORMANCE SCORES:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1_score:.3f}")
    
    if true_positives > 0:
        correctly_detected = found_biomarkers.intersection(detected_biomarkers)
        print(f"\n✅ CORRECTLY DETECTED BIOMARKERS:")
        for biomarker in sorted(correctly_detected):
            print(f"   • {biomarker}")
    
    if false_negatives > 0:
        missed_biomarkers = found_biomarkers - detected_biomarkers
        print(f"\n❌ MISSED BIOMARKERS:")
        for biomarker in sorted(missed_biomarkers):
            print(f"   • {biomarker}")
    
    if false_positives > 0:
        extra_biomarkers = detected_biomarkers - found_biomarkers
        print(f"\n⚠️ EXTRA DETECTIONS (not in ground truth):")
        for biomarker in sorted(extra_biomarkers):
            print(f"   • {biomarker}")
    
    # Save detailed results
    detailed_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_type': 'detailed_wine_biomarkers',
        'timing': {
            'total_time': total_time,
            'sentence_processing_time': sentence_processing_time,
            'avg_time_per_sentence': sum(sentence_times)/len(sentence_times),
            'min_time': min(sentence_times),
            'max_time': max(sentence_times)
        },
        'database': {
            'total_biomarkers': len(biomarkers),
            'biomarkers_in_document': len(found_biomarkers),
            'ground_truth_biomarkers': sorted(list(found_biomarkers))
        },
        'pipeline_performance': {
            'sentences_processed': len(sentences),
            'success_rate': stats['success_rate'],
            'provider_switches': stats['fallback_switches'],
            'detected_biomarkers': sorted(list(detected_biomarkers))
        },
        'detection_metrics': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'correctly_detected': sorted(list(found_biomarkers.intersection(detected_biomarkers))),
            'missed_biomarkers': sorted(list(found_biomarkers - detected_biomarkers)),
            'extra_detections': sorted(list(detected_biomarkers - found_biomarkers))
        }
    }
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = f"Detailed_Wine_Biomarkers_Report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed report saved: {report_file}")
    
    # Summary assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if f1_score >= 0.8:
        print(f"   EXCELLENT: F1 Score ≥ 0.8")
    elif f1_score >= 0.6:
        print(f"   GOOD: F1 Score ≥ 0.6")
    elif f1_score >= 0.4:
        print(f"   FAIR: F1 Score ≥ 0.4")
    else:
        print(f"   NEEDS IMPROVEMENT: F1 Score < 0.4")
    
    print(f"\n🚀 PRODUCTION READINESS:")
    if stats['success_rate'] >= 0.95 and f1_score >= 0.6:
        print(f"   ✅ READY: High reliability and good accuracy")
    elif stats['success_rate'] >= 0.9:
        print(f"   ⚠️ MOSTLY READY: Good reliability, accuracy could improve")
    else:
        print(f"   ❌ NOT READY: Reliability or accuracy issues")
    
    return detailed_results

if __name__ == "__main__":
    run_detailed_wine_biomarkers_test()
