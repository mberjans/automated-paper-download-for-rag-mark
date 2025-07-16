#!/usr/bin/env python3
"""
Test Improved Document-Only Extraction
Test the refined document-only prompt that should extract compounds from text
"""

import sys
sys.path.append('FOODB_LLM_pipeline')

def test_improved_document_only_extraction():
    """Test the improved document-only extraction"""
    print("🔬 Testing Improved Document-Only Extraction")
    print("=" * 45)
    
    from llm_wrapper_enhanced import LLMWrapper
    
    # Test samples with known compounds
    test_samples = [
        {
            'text': "The main urinary biomarkers identified were malvidin-3-glucoside, caffeic acid ethyl ester, and quercetin-3-glucuronide.",
            'expected': ['malvidin-3-glucoside', 'caffeic acid ethyl ester', 'quercetin-3-glucuronide']
        },
        {
            'text': "Analysis revealed significant levels of resveratrol and its metabolites in urine samples.",
            'expected': ['resveratrol']
        },
        {
            'text': "Phenolic compounds including gallic acid, protocatechuic acid, and vanillic acid were detected.",
            'expected': ['gallic acid', 'protocatechuic acid', 'vanillic acid']
        },
        {
            'text': "The study identified anthocyanins such as cyanidin-3-glucoside and peonidin-3-glucoside as key biomarkers.",
            'expected': ['cyanidin-3-glucoside', 'peonidin-3-glucoside']
        },
        {
            'text': "Mass spectrometry analysis detected various sulfate and glucuronide conjugates of wine polyphenols.",
            'expected': []  # No specific compounds mentioned
        }
    ]
    
    # Initialize wrapper
    wrapper = LLMWrapper(document_only_mode=True)
    
    print(f"\n🧪 Testing {len(test_samples)} samples...")
    
    results = []
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nSample {i}: ", end="", flush=True)
        
        # Extract using improved document-only method
        extraction_result = wrapper.extract_metabolites_document_only(sample['text'], 150)
        
        # Parse extracted compounds
        extracted_lines = [line.strip() for line in extraction_result.split('\n') if line.strip()]
        extracted_compounds = []
        
        for line in extracted_lines:
            # Skip common non-compound responses
            if line.lower() not in ['no compounds found', 'no specific compounds mentioned']:
                # Remove numbering if present
                clean_line = line
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-')):
                    clean_line = line[2:].strip()
                if clean_line:
                    extracted_compounds.append(clean_line.lower())
        
        # Compare with expected
        expected_lower = [comp.lower() for comp in sample['expected']]
        matches = sum(1 for comp in extracted_compounds if any(exp in comp or comp in exp for exp in expected_lower))
        
        results.append({
            'sample_id': i,
            'text': sample['text'],
            'expected': sample['expected'],
            'extracted': extracted_compounds,
            'matches': matches,
            'extraction_result': extraction_result
        })
        
        print(f"✅ {len(extracted_compounds)} extracted, {matches} matches")
    
    return results

def analyze_improved_results(results):
    """Analyze the improved extraction results"""
    print("\n📊 Analyzing Improved Extraction Results")
    print("=" * 40)
    
    total_samples = len(results)
    total_expected = sum(len(r['expected']) for r in results)
    total_extracted = sum(len(r['extracted']) for r in results)
    total_matches = sum(r['matches'] for r in results)
    
    print(f"📈 Overall Statistics:")
    print(f"   Samples tested: {total_samples}")
    print(f"   Expected compounds: {total_expected}")
    print(f"   Extracted compounds: {total_extracted}")
    print(f"   Matches: {total_matches}")
    print(f"   Precision: {total_matches/total_extracted:.1%}" if total_extracted > 0 else "   Precision: N/A")
    print(f"   Recall: {total_matches/total_expected:.1%}" if total_expected > 0 else "   Recall: N/A")
    
    print(f"\n📋 Sample-by-Sample Results:")
    for result in results:
        print(f"\n   Sample {result['sample_id']}:")
        print(f"     Text: {result['text'][:80]}...")
        print(f"     Expected: {result['expected']}")
        print(f"     Extracted: {result['extracted']}")
        print(f"     Matches: {result['matches']}/{len(result['expected'])}")
        
        # Show raw extraction for debugging
        if result['extraction_result']:
            print(f"     Raw result: {result['extraction_result'][:100]}...")
    
    return {
        'total_samples': total_samples,
        'total_expected': total_expected,
        'total_extracted': total_extracted,
        'total_matches': total_matches,
        'precision': total_matches/total_extracted if total_extracted > 0 else 0,
        'recall': total_matches/total_expected if total_expected > 0 else 0
    }

def test_contamination_prevention():
    """Test that the improved prompt prevents training data contamination"""
    print("\n🛡️ Testing Training Data Contamination Prevention")
    print("=" * 50)
    
    from llm_wrapper_enhanced import LLMWrapper
    
    # Test with ambiguous text that might trigger training data use
    contamination_tests = [
        {
            'text': "Wine consumption affects metabolism.",
            'description': "Vague text about wine - should not extract specific compounds"
        },
        {
            'text': "The study analyzed urine samples after wine intake.",
            'description': "General study description - should not add known wine metabolites"
        },
        {
            'text': "Polyphenols are important compounds in wine.",
            'description': "General statement - should not list specific polyphenols"
        }
    ]
    
    wrapper = LLMWrapper(document_only_mode=True)
    
    print(f"🧪 Testing {len(contamination_tests)} contamination scenarios...")
    
    contamination_results = []
    
    for i, test in enumerate(contamination_tests, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"   Text: {test['text']}")
        
        result = wrapper.extract_metabolites_document_only(test['text'], 150)
        
        # Check for contamination indicators
        contamination_detected = any(indicator in result.lower() for indicator in [
            'malvidin', 'resveratrol', 'quercetin', 'anthocyanin', 'catechin',
            'gallic acid', 'caffeic acid', 'protocatechuic'
        ])
        
        contamination_results.append({
            'test_id': i,
            'text': test['text'],
            'result': result,
            'contamination_detected': contamination_detected
        })
        
        print(f"   Result: {result}")
        print(f"   Contamination: {'❌ DETECTED' if contamination_detected else '✅ CLEAN'}")
    
    clean_tests = sum(1 for r in contamination_results if not r['contamination_detected'])
    print(f"\n📊 Contamination Prevention Results:")
    print(f"   Clean tests: {clean_tests}/{len(contamination_tests)}")
    print(f"   Success rate: {clean_tests/len(contamination_tests):.1%}")
    
    return contamination_results

def main():
    """Test improved document-only extraction"""
    print("🔬 FOODB Pipeline - Improved Document-Only Extraction Test")
    print("=" * 65)
    
    try:
        # Test improved extraction
        results = test_improved_document_only_extraction()
        
        # Analyze results
        analysis = analyze_improved_results(results)
        
        # Test contamination prevention
        contamination_results = test_contamination_prevention()
        
        print(f"\n🎯 IMPROVED EXTRACTION SUMMARY:")
        print(f"   Precision: {analysis['precision']:.1%}")
        print(f"   Recall: {analysis['recall']:.1%}")
        print(f"   Total extracted: {analysis['total_extracted']} compounds")
        print(f"   Contamination prevention: {'✅ Working' if all(not r['contamination_detected'] for r in contamination_results) else '⚠️ Needs improvement'}")
        
        print(f"\n💡 ASSESSMENT:")
        if analysis['recall'] > 0.7 and analysis['precision'] > 0.7:
            print(f"   ✅ Improved document-only extraction is working well")
            print(f"   ✅ Ready for production deployment")
        elif analysis['recall'] > 0.5:
            print(f"   ⚠️ Moderate performance - may need prompt refinement")
            print(f"   🔧 Consider adjusting extraction instructions")
        else:
            print(f"   ❌ Low performance - prompt needs significant improvement")
            print(f"   🔧 Requires prompt engineering optimization")
        
    except Exception as e:
        print(f"❌ Improved extraction test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
