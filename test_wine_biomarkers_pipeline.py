#!/usr/bin/env python3
"""
Wine Biomarkers Pipeline Testing
Test the FOODB pipeline specifically with the urinary wine biomarkers dataset
"""

import time
import json
import sys
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Add pipeline directory to path
sys.path.append('FOODB_LLM_pipeline')

def load_wine_biomarkers_database():
    """Load the wine biomarkers CSV file"""
    print("🍷 Loading Wine Biomarkers Database")
    print("=" * 35)
    
    csv_file = "urinary_wine_biomarkers.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded wine biomarkers database")
        print(f"   File: {csv_file}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Show sample compounds
        if 'Compound Name' in df.columns:
            sample_compounds = df['Compound Name'].head(10).tolist()
            print(f"   Sample compounds: {sample_compounds}")
        
        return {
            'file': csv_file,
            'data': df,
            'compound_count': len(df)
        }
        
    except Exception as e:
        print(f"❌ Error loading {csv_file}: {e}")
        return None

def create_wine_research_document():
    """Create a realistic wine research document with known biomarkers"""
    print("\n📝 Creating Wine Research Document")
    print("=" * 35)
    
    # Create a scientific document about wine biomarkers
    wine_document = """
    Urinary Biomarkers of Wine Consumption: A Comprehensive Analysis
    
    Abstract:
    Wine consumption can be monitored through the detection of specific urinary biomarkers. 
    This study examines the excretion patterns of wine-derived compounds in human urine 
    following controlled wine consumption.
    
    Introduction:
    Red wine contains numerous polyphenolic compounds that are metabolized and excreted 
    in urine. These compounds serve as reliable biomarkers for wine intake assessment.
    
    Phenolic Acids and Derivatives:
    Tartaric acid is a major organic acid in wine that appears unchanged in urine after 
    consumption. Gallic acid, a phenolic acid abundant in red wine, is also detected 
    in urine samples.
    
    Hippuric acid levels increase significantly after wine consumption, likely due to 
    the metabolism of benzoic acid derivatives present in wine.
    
    Flavonoid Metabolites:
    Quercetin, a major flavonol in wine, is metabolized to various conjugated forms 
    that appear in urine. Catechin and epicatechin from wine are also excreted as 
    glucuronide and sulfate conjugates.
    
    Resveratrol, the famous stilbene compound in red wine, appears in urine primarily 
    as resveratrol-3-O-glucuronide and resveratrol-4'-O-glucuronide.
    
    Anthocyanin Metabolites:
    Malvidin, the predominant anthocyanin in red wine, is detected in urine as 
    malvidin-3-O-glucoside and its metabolites. Peonidin and cyanidin derivatives 
    are also found in post-consumption urine samples.
    
    Organic Acids:
    Citric acid concentrations in urine correlate with wine consumption patterns. 
    Malic acid, another wine acid, shows increased urinary excretion after wine intake.
    
    Caffeic acid and its metabolites, including caffeic acid-3-O-sulfate, are 
    reliable indicators of wine consumption.
    
    Hydroxybenzoic Acid Derivatives:
    4-Hydroxybenzoic acid and vanillic acid levels in urine increase following 
    wine consumption. These compounds originate from the breakdown of more complex 
    wine phenolics.
    
    Protocatechuic acid, a metabolite of anthocyanins and other phenolics, is 
    consistently detected in post-wine consumption urine samples.
    
    Novel Biomarkers:
    Syringic acid has emerged as a specific biomarker for red wine consumption. 
    Homovanillic acid levels also correlate with wine intake.
    
    Ferulic acid and its conjugates show increased urinary excretion after wine 
    consumption, making them useful biomarkers.
    
    Methodology Considerations:
    Urine collection timing is critical, as most wine biomarkers peak 2-6 hours 
    post-consumption. Sample preservation and analytical methods significantly 
    impact biomarker detection.
    
    Conclusion:
    Multiple urinary biomarkers can reliably indicate wine consumption. The combination 
    of tartaric acid, resveratrol metabolites, and specific phenolic acids provides 
    robust evidence of wine intake.
    """
    
    # Save document
    os.makedirs("FOODB_LLM_pipeline/sample_input", exist_ok=True)
    doc_file = "FOODB_LLM_pipeline/sample_input/wine_biomarkers_research.txt"
    
    with open(doc_file, 'w') as f:
        f.write(wine_document)
    
    print(f"✅ Created wine research document: {doc_file}")
    print(f"📄 Document length: {len(wine_document)} characters")
    
    return doc_file, wine_document

def extract_wine_biomarkers_from_text(text: str, biomarkers_db: Dict) -> Tuple[Set[str], Dict]:
    """Extract wine biomarkers from text using the database"""
    print("\n🔬 Extracting Wine Biomarkers from Text")
    print("=" * 40)
    
    start_time = time.time()
    
    # Get compound names from database
    df = biomarkers_db['data']
    
    if 'Compound Name' in df.columns:
        biomarker_names = set(df['Compound Name'].str.lower().tolist())
    else:
        print("❌ No 'Compound Name' column found")
        return set(), {}
    
    # Clean up compound names (remove extra spaces, etc.)
    biomarker_names = {name.strip() for name in biomarker_names if isinstance(name, str)}
    
    # Extract biomarkers found in text
    text_lower = text.lower()
    found_biomarkers = set()
    biomarker_positions = {}
    
    for biomarker in biomarker_names:
        if len(biomarker) > 3 and biomarker in text_lower:  # Avoid very short matches
            found_biomarkers.add(biomarker)
            # Find positions
            positions = []
            start = 0
            while True:
                pos = text_lower.find(biomarker, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            biomarker_positions[biomarker] = positions
    
    end_time = time.time()
    extraction_time = end_time - start_time
    
    print(f"⏱️ Extraction time: {extraction_time:.3f}s")
    print(f"📊 Database biomarkers: {len(biomarker_names)}")
    print(f"✅ Found biomarkers: {len(found_biomarkers)}")
    
    if found_biomarkers:
        print(f"🎯 Found biomarkers: {sorted(list(found_biomarkers))}")
    
    return found_biomarkers, {
        'extraction_time': extraction_time,
        'database_size': len(biomarker_names),
        'found_count': len(found_biomarkers),
        'found_biomarkers': list(found_biomarkers),
        'positions': biomarker_positions
    }

def run_wine_pipeline_analysis(doc_file: str) -> Tuple[List[Dict], Dict]:
    """Run the complete pipeline analysis on wine document"""
    print("\n🧪 Running Wine Pipeline Analysis")
    print("=" * 35)
    
    start_time = time.time()
    
    try:
        from llm_wrapper_enhanced import LLMWrapper
        
        wrapper = LLMWrapper()
        print(f"🎯 Primary provider: {wrapper.current_provider}")
        
        # Read document
        with open(doc_file, 'r') as f:
            text = f.read()
        
        # Split into meaningful chunks (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        print(f"📝 Processing {len(paragraphs)} paragraphs...")
        
        processed_results = []
        processing_times = []
        
        for i, paragraph in enumerate(paragraphs, 1):
            print(f"  Paragraph {i}: ", end="", flush=True)
            
            # Create specific prompt for wine biomarker extraction
            prompt = f"""Extract wine biomarkers and metabolites from this text about wine consumption:

{paragraph}

List all compounds, metabolites, and biomarkers mentioned that could be found in urine after wine consumption."""
            
            para_start = time.time()
            response = wrapper.generate_single(prompt, max_tokens=200)
            para_end = time.time()
            
            para_time = para_end - para_start
            processing_times.append(para_time)
            
            if response:
                processed_results.append({
                    'paragraph_number': i,
                    'original_text': paragraph,
                    'extracted_biomarkers': response,
                    'processing_time': para_time
                })
                print(f"✅ {para_time:.2f}s")
            else:
                print(f"❌ {para_time:.2f}s")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get wrapper statistics
        stats = wrapper.get_statistics()
        
        print(f"\n📊 Wine Pipeline Analysis Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Processed paragraphs: {len(processed_results)}")
        print(f"  Average time per paragraph: {sum(processing_times)/len(processing_times):.2f}s")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Provider switches: {stats['fallback_switches']}")
        
        return processed_results, {
            'total_time': total_time,
            'paragraph_count': len(processed_results),
            'avg_time_per_paragraph': sum(processing_times)/len(processing_times) if processing_times else 0,
            'success_rate': stats['success_rate'],
            'provider_switches': stats['fallback_switches'],
            'processing_times': processing_times,
            'wrapper_stats': stats
        }
        
    except Exception as e:
        print(f"❌ Error in wine pipeline analysis: {e}")
        import traceback
        traceback.print_exc()
        return [], {'total_time': 0, 'error': str(e)}

def calculate_wine_biomarker_metrics(found_biomarkers: Set[str], pipeline_results: List[Dict], 
                                   biomarkers_db: Dict) -> Dict:
    """Calculate precision, recall, F1 for wine biomarker detection"""
    print("\n📊 Calculating Wine Biomarker Detection Metrics")
    print("=" * 50)
    
    # Extract biomarkers mentioned in pipeline results
    pipeline_biomarkers = set()
    
    # Get all biomarker names from database for matching
    df = biomarkers_db['data']
    db_biomarkers = set(df['Compound Name'].str.lower().tolist())
    
    # Look for biomarkers in pipeline responses
    for result in pipeline_results:
        extracted_text = result.get('extracted_biomarkers', '').lower()
        
        # Check each database biomarker
        for biomarker in db_biomarkers:
            if len(biomarker) > 3 and biomarker in extracted_text:
                pipeline_biomarkers.add(biomarker)
    
    # Calculate metrics
    true_positives = len(found_biomarkers.intersection(pipeline_biomarkers))
    false_positives = len(pipeline_biomarkers - found_biomarkers)
    false_negatives = len(found_biomarkers - pipeline_biomarkers)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 Wine Biomarker Detection Metrics:")
    print(f"  Ground truth biomarkers: {len(found_biomarkers)}")
    print(f"  Pipeline detected biomarkers: {len(pipeline_biomarkers)}")
    print(f"  True positives: {true_positives}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1_score:.3f}")
    
    # Show specific matches
    if true_positives > 0:
        matched_biomarkers = found_biomarkers.intersection(pipeline_biomarkers)
        print(f"  ✅ Correctly detected: {sorted(list(matched_biomarkers))}")
    
    if false_negatives > 0:
        missed_biomarkers = found_biomarkers - pipeline_biomarkers
        print(f"  ❌ Missed biomarkers: {sorted(list(missed_biomarkers))}")
    
    return {
        'ground_truth_count': len(found_biomarkers),
        'pipeline_detected_count': len(pipeline_biomarkers),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ground_truth_biomarkers': sorted(list(found_biomarkers)),
        'pipeline_biomarkers': sorted(list(pipeline_biomarkers)),
        'correctly_detected': sorted(list(found_biomarkers.intersection(pipeline_biomarkers))),
        'missed_biomarkers': sorted(list(found_biomarkers - pipeline_biomarkers))
    }

def save_wine_biomarker_report(all_metrics: Dict):
    """Save wine biomarker analysis report"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = f"Wine_Biomarkers_Pipeline_Report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n💾 Wine biomarker report saved: {report_file}")
    return report_file

def main():
    """Run comprehensive wine biomarkers pipeline testing"""
    print("🍷 FOODB Pipeline - Wine Biomarkers Testing")
    print("=" * 50)
    
    overall_start = time.time()
    
    try:
        # Step 1: Load wine biomarkers database
        step1_start = time.time()
        biomarkers_db = load_wine_biomarkers_database()
        step1_end = time.time()
        
        if not biomarkers_db:
            print("❌ Could not load wine biomarkers database")
            return
        
        # Step 2: Create wine research document
        step2_start = time.time()
        doc_file, wine_text = create_wine_research_document()
        step2_end = time.time()
        
        # Step 3: Extract ground truth biomarkers
        step3_start = time.time()
        found_biomarkers, extraction_metrics = extract_wine_biomarkers_from_text(wine_text, biomarkers_db)
        step3_end = time.time()
        
        # Step 4: Run pipeline analysis
        step4_start = time.time()
        pipeline_results, pipeline_metrics = run_wine_pipeline_analysis(doc_file)
        step4_end = time.time()
        
        # Step 5: Calculate detection metrics
        step5_start = time.time()
        detection_metrics = calculate_wine_biomarker_metrics(found_biomarkers, pipeline_results, biomarkers_db)
        step5_end = time.time()
        
        overall_end = time.time()
        total_time = overall_end - overall_start
        
        # Compile comprehensive metrics
        comprehensive_metrics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'wine_biomarkers_pipeline',
            'total_processing_time': total_time,
            'step_times': {
                'database_loading': step1_end - step1_start,
                'document_creation': step2_end - step2_start,
                'ground_truth_extraction': step3_end - step3_start,
                'pipeline_analysis': step4_end - step4_start,
                'metrics_calculation': step5_end - step5_start
            },
            'wine_biomarkers_database': {
                'file': biomarkers_db['file'],
                'compound_count': biomarkers_db['compound_count']
            },
            'ground_truth_metrics': extraction_metrics,
            'pipeline_metrics': pipeline_metrics,
            'detection_performance': detection_metrics
        }
        
        # Display final results
        print(f"\n🎉 WINE BIOMARKERS PIPELINE TESTING COMPLETE!")
        print("=" * 55)
        
        print(f"\n⏱️ TIMING BREAKDOWN:")
        for step, duration in comprehensive_metrics['step_times'].items():
            print(f"  {step.replace('_', ' ').title()}: {duration:.3f}s")
        print(f"  TOTAL PROCESSING TIME: {total_time:.3f}s")
        
        print(f"\n🍷 WINE BIOMARKER DETECTION:")
        print(f"  CSV Database Size: {biomarkers_db['compound_count']} compounds")
        print(f"  Found in Document: {detection_metrics['ground_truth_count']} biomarkers")
        print(f"  Detected by Pipeline: {detection_metrics['pipeline_detected_count']} biomarkers")
        print(f"  True Positives: {detection_metrics['true_positives']}")
        
        print(f"\n🎯 PERFORMANCE SCORES:")
        print(f"  Precision: {detection_metrics['precision']:.3f}")
        print(f"  Recall: {detection_metrics['recall']:.3f}")
        print(f"  F1 Score: {detection_metrics['f1_score']:.3f}")
        
        print(f"\n⚡ PIPELINE PERFORMANCE:")
        print(f"  Processing Speed: {pipeline_metrics.get('avg_time_per_paragraph', 0):.3f}s/paragraph")
        print(f"  Success Rate: {pipeline_metrics.get('success_rate', 0):.1%}")
        print(f"  Provider Switches: {pipeline_metrics.get('provider_switches', 0)}")
        
        # Save comprehensive report
        save_wine_biomarker_report(comprehensive_metrics)
        
        return comprehensive_metrics
        
    except Exception as e:
        print(f"❌ Wine biomarkers pipeline testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
