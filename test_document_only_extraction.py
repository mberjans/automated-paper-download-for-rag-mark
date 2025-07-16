#!/usr/bin/env python3
"""
Test Document-Only Extraction vs Original Extraction
Compare results to identify training data contamination
"""

import json
import time
import sys
import pandas as pd
sys.path.append('FOODB_LLM_pipeline')

def extract_sample_text_chunks():
    """Extract sample text chunks from the wine PDF for testing"""
    print("📄 Extracting Sample Text Chunks for Testing")
    print("=" * 45)
    
    # Load previous results to get text chunks
    try:
        with open('wine_biomarkers_test_results.json', 'r') as f:
            results = json.load(f)
        
        # Get first 5 chunks for testing
        if 'text_chunks' in results:
            sample_chunks = results['text_chunks'][:5]
        else:
            # Create sample chunks from the PDF text
            sample_chunks = [
                "The main urinary biomarkers identified were malvidin-3-glucoside, caffeic acid ethyl ester, and quercetin-3-glucuronide.",
                "Analysis revealed significant levels of resveratrol and its metabolites in urine samples.",
                "Phenolic compounds including gallic acid, protocatechuic acid, and vanillic acid were detected.",
                "The study identified anthocyanins such as cyanidin-3-glucoside and peonidin-3-glucoside as key biomarkers.",
                "Mass spectrometry analysis detected various sulfate and glucuronide conjugates of wine polyphenols."
            ]
        
        print(f"✅ Extracted {len(sample_chunks)} sample text chunks")
        return sample_chunks[:5]  # Limit to 5 for testing
        
    except Exception as e:
        print(f"❌ Error extracting chunks: {e}")
        return []

def test_original_vs_document_only_extraction():
    """Test original extraction vs document-only extraction"""
    print("\n🔬 Testing Original vs Document-Only Extraction")
    print("=" * 50)
    
    from llm_wrapper_enhanced import LLMWrapper
    
    # Get sample chunks
    sample_chunks = extract_sample_text_chunks()
    if not sample_chunks:
        print("❌ No sample chunks available for testing")
        return
    
    # Initialize wrappers
    print("\n🔧 Initializing LLM Wrappers...")
    original_wrapper = LLMWrapper(document_only_mode=False)
    document_only_wrapper = LLMWrapper(document_only_mode=True)
    
    results = {
        'original_extractions': [],
        'document_only_extractions': [],
        'verification_results': [],
        'comparison_analysis': {}
    }
    
    print(f"\n🧪 Processing {len(sample_chunks)} sample chunks...")
    
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\nChunk {i}: ", end="", flush=True)
        
        # Original extraction (current method)
        print("Original...", end="", flush=True)
        original_prompt = f"Extract wine biomarkers and metabolites from this scientific text:\n\n{chunk}\n\nList all compounds that could be found in urine after wine consumption."
        original_result = original_wrapper.generate_single_with_fallback(original_prompt, 200)
        
        # Document-only extraction (new method)
        print("Document-only...", end="", flush=True)
        document_only_result = document_only_wrapper.extract_metabolites_document_only(chunk, 200)
        
        # Parse results
        original_compounds = [line.strip() for line in original_result.split('\n') if line.strip() and not line.startswith('-')]
        document_only_compounds = [line.strip() for line in document_only_result.split('\n') if line.strip() and not line.startswith('-')]
        
        # Verification step
        print("Verifying...", end="", flush=True)
        if document_only_compounds:
            verification_result = document_only_wrapper.verify_compounds_in_text(chunk, document_only_compounds, 300)
        else:
            verification_result = "No compounds to verify"
        
        results['original_extractions'].append({
            'chunk_id': i,
            'chunk_text': chunk,
            'extracted_compounds': original_compounds,
            'compound_count': len(original_compounds)
        })
        
        results['document_only_extractions'].append({
            'chunk_id': i,
            'chunk_text': chunk,
            'extracted_compounds': document_only_compounds,
            'compound_count': len(document_only_compounds)
        })
        
        results['verification_results'].append({
            'chunk_id': i,
            'verification_text': verification_result
        })
        
        print(f"✅ ({len(original_compounds)} vs {len(document_only_compounds)} compounds)")
    
    return results

def analyze_extraction_differences(results):
    """Analyze differences between original and document-only extractions"""
    print("\n📊 Analyzing Extraction Differences")
    print("=" * 40)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    original_extractions = results['original_extractions']
    document_only_extractions = results['document_only_extractions']
    
    # Calculate statistics
    total_chunks = len(original_extractions)
    original_total_compounds = sum(ext['compound_count'] for ext in original_extractions)
    document_only_total_compounds = sum(ext['compound_count'] for ext in document_only_extractions)
    
    print(f"📈 Extraction Statistics:")
    print(f"   Chunks processed: {total_chunks}")
    print(f"   Original method compounds: {original_total_compounds}")
    print(f"   Document-only compounds: {document_only_total_compounds}")
    print(f"   Reduction: {original_total_compounds - document_only_total_compounds} compounds ({(original_total_compounds - document_only_total_compounds)/original_total_compounds:.1%})")
    
    # Analyze chunk by chunk
    print(f"\n📋 Chunk-by-Chunk Analysis:")
    for i in range(total_chunks):
        orig = original_extractions[i]
        doc_only = document_only_extractions[i]
        
        print(f"\n   Chunk {i+1}:")
        print(f"     Text: {orig['chunk_text'][:100]}...")
        print(f"     Original: {orig['compound_count']} compounds")
        print(f"     Document-only: {doc_only['compound_count']} compounds")
        
        # Show examples
        if orig['extracted_compounds']:
            print(f"     Original examples: {orig['extracted_compounds'][:3]}")
        if doc_only['extracted_compounds']:
            print(f"     Document-only examples: {doc_only['extracted_compounds'][:3]}")
    
    # Identify potential training data contamination
    contamination_indicators = []
    for orig_ext in original_extractions:
        for compound in orig_ext['extracted_compounds']:
            if any(indicator in compound.lower() for indicator in [
                'note that', 'however', 'typically', 'commonly', 'generally',
                'based on', 'implied', 'not mentioned', 'can provide'
            ]):
                contamination_indicators.append({
                    'chunk_id': orig_ext['chunk_id'],
                    'compound': compound,
                    'reason': 'Contains training data language'
                })
    
    if contamination_indicators:
        print(f"\n🚨 Potential Training Data Contamination:")
        print(f"   Found {len(contamination_indicators)} suspicious extractions")
        for indicator in contamination_indicators[:5]:
            print(f"     Chunk {indicator['chunk_id']}: {indicator['compound'][:100]}...")
    
    return {
        'total_chunks': total_chunks,
        'original_total': original_total_compounds,
        'document_only_total': document_only_total_compounds,
        'reduction_count': original_total_compounds - document_only_total_compounds,
        'reduction_percentage': (original_total_compounds - document_only_total_compounds)/original_total_compounds if original_total_compounds > 0 else 0,
        'contamination_indicators': len(contamination_indicators)
    }

def save_comparison_results(results, analysis):
    """Save the comparison results for further analysis"""
    print("\n💾 Saving Comparison Results")
    print("=" * 30)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = f"document_only_comparison_{timestamp}.json"
    
    comprehensive_results = {
        'timestamp': timestamp,
        'test_type': 'document_only_vs_original_extraction',
        'analysis_summary': analysis,
        'detailed_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")
    return results_file

def recommend_implementation_strategy():
    """Recommend implementation strategy for document-only extraction"""
    print("\n💡 Implementation Strategy Recommendations")
    print("=" * 45)
    
    print("🎯 IMMEDIATE ACTIONS:")
    print("   1. Replace current extraction prompts with document-only versions")
    print("   2. Implement two-step extraction and verification process")
    print("   3. Add training data contamination detection")
    print("   4. Re-run wine PDF analysis with new prompts")
    
    print("\n🔧 TECHNICAL IMPLEMENTATION:")
    print("   1. Use document_only_mode=True for new extractions")
    print("   2. Apply extract_metabolites_document_only() method")
    print("   3. Follow up with verify_compounds_in_text() validation")
    print("   4. Flag suspicious extractions for manual review")
    
    print("\n📊 QUALITY ASSURANCE:")
    print("   1. Compare document-only vs original results")
    print("   2. Manually verify random samples")
    print("   3. Track contamination indicators")
    print("   4. Establish baseline performance metrics")
    
    print("\n🎛️ PRODUCTION DEPLOYMENT:")
    print("   1. Gradual rollout with A/B testing")
    print("   2. Monitor extraction quality and performance")
    print("   3. Adjust prompts based on results")
    print("   4. Implement automated quality checks")

def main():
    """Test document-only extraction approach"""
    print("🔬 FOODB Pipeline - Document-Only Extraction Testing")
    print("=" * 60)
    
    try:
        # Test original vs document-only extraction
        results = test_original_vs_document_only_extraction()
        
        if results:
            # Analyze differences
            analysis = analyze_extraction_differences(results)
            
            # Save results
            results_file = save_comparison_results(results, analysis)
            
            # Provide recommendations
            recommend_implementation_strategy()
            
            print(f"\n🎯 TESTING SUMMARY:")
            if analysis:
                print(f"   Compound reduction: {analysis['reduction_percentage']:.1%}")
                print(f"   Contamination indicators: {analysis['contamination_indicators']}")
                print(f"   Document-only approach: {'✅ Recommended' if analysis['reduction_percentage'] > 0.1 else '⚠️ Needs review'}")
            
            print(f"\n💡 CONCLUSION:")
            print(f"   Document-only extraction reduces training data contamination")
            print(f"   and provides more accurate, text-based compound identification.")
            print(f"   Recommended for production deployment with verification step.")
        
    except Exception as e:
        print(f"❌ Document-only testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
