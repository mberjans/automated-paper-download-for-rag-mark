#!/usr/bin/env python3
"""
Investigate Detection Rate Discrepancy
Compare previous high detection rates with current 57.6% rate
and analyze metabolites detected outside CSV file
"""

import json
import pandas as pd
from typing import Set, List, Dict

def load_previous_results():
    """Load previous test results that showed high detection rates"""
    print("📊 Loading Previous Test Results")
    print("=" * 35)
    
    try:
        # Try to load the most recent comprehensive results
        with open('wine_biomarkers_test_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"✅ Loaded previous results")
        return results
    except Exception as e:
        print(f"❌ Error loading previous results: {e}")
        return None

def load_csv_biomarkers():
    """Load the CSV biomarkers database"""
    print("\n📋 Loading CSV Biomarkers Database")
    print("=" * 35)
    
    try:
        df = pd.read_csv("urinary_wine_biomarkers.csv")
        csv_biomarkers = set(df['Compound Name'].str.lower().str.strip().tolist())
        
        print(f"✅ Loaded {len(csv_biomarkers)} biomarkers from CSV")
        
        # Show some examples
        print(f"\n📋 Sample CSV biomarkers:")
        for i, biomarker in enumerate(sorted(list(csv_biomarkers))[:10], 1):
            print(f"   {i:2d}. {biomarker}")
        
        return csv_biomarkers
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return set()

def analyze_previous_detection_claims():
    """Analyze the previous claims of high detection rates"""
    print("\n🔍 Analyzing Previous Detection Claims")
    print("=" * 40)
    
    # From previous test results, we claimed:
    previous_claims = {
        'test_1': {
            'description': 'Initial wine PDF test (with provider switching)',
            'claimed_matches': 32,
            'total_biomarkers': 59,
            'claimed_recall': 32/59,
            'chunks_processed': 16
        },
        'test_2': {
            'description': 'Complete wine PDF test (with exponential backoff)', 
            'claimed_matches': 57,
            'total_biomarkers': 59,
            'claimed_recall': 57/59,
            'chunks_processed': 45
        }
    }
    
    print(f"📊 Previous Test Claims:")
    for test_id, data in previous_claims.items():
        print(f"\n{test_id.upper()}: {data['description']}")
        print(f"   Claimed matches: {data['claimed_matches']}/{data['total_biomarkers']}")
        print(f"   Claimed recall: {data['claimed_recall']:.1%}")
        print(f"   Chunks processed: {data['chunks_processed']}")
    
    return previous_claims

def investigate_matching_methodology():
    """Investigate how the matching was done previously vs now"""
    print("\n🔬 Investigating Matching Methodology")
    print("=" * 40)
    
    print(f"🔍 PREVIOUS MATCHING APPROACH (from test output):")
    print(f"   The previous test showed matches like:")
    print(f"   • Expected: Malvidin-3-glucoside → Extracted: Glucoside")
    print(f"   • Expected: Caffeic acid ethyl ester → Extracted: Caffeic acid")
    print(f"   • Expected: Peonidin-3-glucuronide → Extracted: Glucuronide")
    
    print(f"\n❓ POTENTIAL ISSUES WITH PREVIOUS MATCHING:")
    print(f"   1. Very loose partial matching (e.g., 'Glucoside' matches many compounds)")
    print(f"   2. Generic terms counted as specific biomarkers")
    print(f"   3. Substring matching without context validation")
    
    print(f"\n🔍 CURRENT MATCHING APPROACH:")
    print(f"   More stringent matching requiring:")
    print(f"   • Exact compound name matches")
    print(f"   • Meaningful partial matches (length > 5 characters)")
    print(f"   • Bidirectional substring validation")
    
    return {
        'previous_approach': 'loose_partial_matching',
        'current_approach': 'stringent_exact_matching',
        'likely_cause': 'previous_approach_was_too_permissive'
    }

def analyze_extracted_metabolites(results):
    """Analyze what metabolites were actually extracted"""
    print("\n🧬 Analyzing Extracted Metabolites")
    print("=" * 35)
    
    if not results or 'extracted_metabolites' not in results:
        print("❌ No extracted metabolites found in results")
        return None
    
    extracted = results['extracted_metabolites']
    print(f"✅ Found {len(extracted)} extracted metabolites")
    
    # Categorize metabolites
    categories = {
        'specific_compounds': [],
        'generic_terms': [],
        'compound_classes': [],
        'descriptive_text': []
    }
    
    # Keywords for categorization
    generic_keywords = ['metabolites', 'compounds', 'biomarkers', 'phenolic', 'anthocyanins']
    class_keywords = ['glucoside', 'glucuronide', 'sulfate', 'acid', 'ester']
    
    for metabolite in extracted:
        metabolite_lower = metabolite.lower()
        
        if any(keyword in metabolite_lower for keyword in generic_keywords):
            categories['generic_terms'].append(metabolite)
        elif any(keyword in metabolite_lower for keyword in class_keywords) and len(metabolite) < 20:
            categories['compound_classes'].append(metabolite)
        elif len(metabolite) > 50:
            categories['descriptive_text'].append(metabolite)
        else:
            categories['specific_compounds'].append(metabolite)
    
    print(f"\n📊 Metabolite Categories:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
        if items:
            print(f"      Examples: {items[:3]}")
    
    return categories

def compare_with_csv_biomarkers(extracted_metabolites, csv_biomarkers):
    """Compare extracted metabolites with CSV biomarkers"""
    print("\n🔍 Comparing with CSV Biomarkers")
    print("=" * 35)
    
    if not extracted_metabolites:
        print("❌ No extracted metabolites to compare")
        return None
    
    # Find matches using different matching strategies
    matching_results = {
        'exact_matches': set(),
        'partial_matches': set(),
        'loose_matches': set(),
        'non_csv_metabolites': set()
    }
    
    extracted_set = set(m.lower().strip() for m in extracted_metabolites)
    
    for extracted in extracted_set:
        found_match = False
        
        # Exact match
        if extracted in csv_biomarkers:
            matching_results['exact_matches'].add(extracted)
            found_match = True
        else:
            # Partial match (both directions)
            for csv_biomarker in csv_biomarkers:
                if len(extracted) > 5 and extracted in csv_biomarker:
                    matching_results['partial_matches'].add(csv_biomarker)
                    found_match = True
                    break
                elif len(csv_biomarker) > 5 and csv_biomarker in extracted:
                    matching_results['partial_matches'].add(csv_biomarker)
                    found_match = True
                    break
        
        # If no match found, it's a non-CSV metabolite
        if not found_match:
            matching_results['non_csv_metabolites'].add(extracted)
    
    # Loose matching (like previous approach)
    for extracted in extracted_set:
        if extracted not in matching_results['exact_matches']:
            for csv_biomarker in csv_biomarkers:
                # Very loose matching (like 'glucoside' matching 'malvidin-3-glucoside')
                if len(extracted) > 3 and (extracted in csv_biomarker or csv_biomarker in extracted):
                    matching_results['loose_matches'].add(csv_biomarker)
                    break
    
    print(f"📊 Matching Results:")
    print(f"   Exact matches: {len(matching_results['exact_matches'])}")
    print(f"   Partial matches: {len(matching_results['partial_matches'])}")
    print(f"   Loose matches: {len(matching_results['loose_matches'])}")
    print(f"   Non-CSV metabolites: {len(matching_results['non_csv_metabolites'])}")
    
    return matching_results

def analyze_non_csv_metabolites(non_csv_metabolites):
    """Analyze metabolites detected that are not in CSV file"""
    print("\n🆕 Analyzing Non-CSV Metabolites")
    print("=" * 35)
    
    if not non_csv_metabolites:
        print("❌ No non-CSV metabolites found")
        return
    
    print(f"✅ Found {len(non_csv_metabolites)} metabolites NOT in CSV file")
    
    # Categorize non-CSV metabolites
    wine_related = []
    generic_terms = []
    specific_compounds = []
    
    wine_keywords = ['wine', 'grape', 'polyphenol', 'anthocyan', 'resveratrol', 'tannin']
    generic_keywords = ['metabolites', 'compounds', 'biomarkers', 'phenolic acids']
    
    for metabolite in non_csv_metabolites:
        metabolite_lower = metabolite.lower()
        
        if any(keyword in metabolite_lower for keyword in wine_keywords):
            wine_related.append(metabolite)
        elif any(keyword in metabolite_lower for keyword in generic_keywords):
            generic_terms.append(metabolite)
        else:
            specific_compounds.append(metabolite)
    
    print(f"\n📊 Non-CSV Metabolite Categories:")
    print(f"   Wine-related: {len(wine_related)}")
    print(f"   Generic terms: {len(generic_terms)}")
    print(f"   Specific compounds: {len(specific_compounds)}")
    
    # Show examples
    if wine_related:
        print(f"\n🍷 Wine-related metabolites (not in CSV):")
        for metabolite in wine_related[:10]:
            print(f"      • {metabolite}")
    
    if specific_compounds:
        print(f"\n🧬 Specific compounds (not in CSV):")
        for metabolite in specific_compounds[:10]:
            print(f"      • {metabolite}")
    
    return {
        'wine_related': wine_related,
        'generic_terms': generic_terms,
        'specific_compounds': specific_compounds
    }

def explain_detection_rate_discrepancy():
    """Explain why detection rate dropped from ~100% to 57.6%"""
    print("\n❓ DETECTION RATE DISCREPANCY EXPLANATION")
    print("=" * 45)
    
    print(f"🔍 LIKELY CAUSES:")
    
    print(f"\n1️⃣ MATCHING METHODOLOGY CHANGE:")
    print(f"   Previous: Very loose partial matching")
    print(f"   • 'Glucoside' matched 'Malvidin-3-glucoside'")
    print(f"   • 'Caffeic acid' matched 'Caffeic acid ethyl ester'")
    print(f"   • Generic terms counted as specific biomarkers")
    print(f"   Current: Stricter exact/meaningful partial matching")
    print(f"   • Requires more precise compound identification")
    
    print(f"\n2️⃣ EXTRACTION QUALITY DIFFERENCE:")
    print(f"   Previous: May have extracted more generic terms")
    print(f"   Current: More specific compound extraction")
    
    print(f"\n3️⃣ EVALUATION CRITERIA:")
    print(f"   Previous: Counted loose matches as successes")
    print(f"   Current: More stringent evaluation criteria")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Re-run analysis with previous loose matching to verify")
    print(f"   Then decide on appropriate matching strictness for production")

def main():
    """Investigate the detection rate discrepancy"""
    print("🔍 FOODB Pipeline - Detection Rate Investigation")
    print("=" * 55)
    
    try:
        # Load data
        results = load_previous_results()
        csv_biomarkers = load_csv_biomarkers()
        
        # Analyze previous claims
        previous_claims = analyze_previous_detection_claims()
        
        # Investigate matching methodology
        methodology = investigate_matching_methodology()
        
        if results:
            # Analyze extracted metabolites
            categories = analyze_extracted_metabolites(results)
            
            # Compare with CSV
            if 'extracted_metabolites' in results:
                matching_results = compare_with_csv_biomarkers(
                    results['extracted_metabolites'], csv_biomarkers
                )
                
                if matching_results:
                    # Analyze non-CSV metabolites
                    analyze_non_csv_metabolites(matching_results['non_csv_metabolites'])
        
        # Explain discrepancy
        explain_detection_rate_discrepancy()
        
        print(f"\n🎯 SUMMARY:")
        print(f"The detection rate discrepancy is likely due to:")
        print(f"1. Previous loose matching methodology")
        print(f"2. Current stricter evaluation criteria")
        print(f"3. Different extraction quality/focus")
        print(f"\nBoth approaches have merit - loose for discovery, strict for precision.")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
