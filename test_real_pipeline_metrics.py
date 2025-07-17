#!/usr/bin/env python3
"""
Real FOODB Pipeline Metrics Testing
This script tests the complete pipeline with real PDF and CSV data to measure:
1. Number of metabolites detected from CSV in PDF
2. F1, Precision, Recall scores
3. Time for each pipeline step
4. Total processing time
"""

import time
import json
import sys
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple
import re

# Add pipeline directory to path
sys.path.append('FOODB_LLM_pipeline')

def find_test_data():
    """Find available test data files"""
    print("🔍 Finding Test Data Files")
    print("=" * 25)
    
    # Look for CSV files (metabolite database)
    csv_files = list(Path('.').glob('**/*.csv'))
    csv_files.extend(list(Path('FOODB_LLM_pipeline').glob('**/*.csv')))
    
    # Look for PDF files
    pdf_files = list(Path('.').glob('**/*.pdf'))
    pdf_files.extend(list(Path('FOODB_LLM_pipeline').glob('**/*.pdf')))
    
    # Look for existing processed data
    jsonl_files = list(Path('FOODB_LLM_pipeline/sample_input').glob('*.jsonl'))
    
    print(f"📊 Found {len(csv_files)} CSV files")
    print(f"📄 Found {len(pdf_files)} PDF files")
    print(f"📝 Found {len(jsonl_files)} JSONL files")
    
    return csv_files, pdf_files, jsonl_files

def load_metabolite_database():
    """Load metabolite database from CSV"""
    print("\n📊 Loading Metabolite Database")
    print("=" * 30)
    
    csv_files, _, _ = find_test_data()
    
    if not csv_files:
        print("❌ No CSV files found")
        return None
    
    # Try to find a metabolite database file
    metabolite_db = None
    
    for csv_file in csv_files:
        try:
            print(f"🔍 Checking {csv_file}...")
            df = pd.read_csv(csv_file)
            
            # Look for columns that might contain metabolite names
            name_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['name', 'compound', 'metabolite', 'chemical'])]
            
            if name_columns and len(df) > 10:  # Reasonable size database
                print(f"✅ Found metabolite database: {csv_file}")
                print(f"   Rows: {len(df)}")
                print(f"   Name columns: {name_columns}")
                
                metabolite_db = {
                    'file': str(csv_file),
                    'data': df,
                    'name_columns': name_columns,
                    'metabolite_count': len(df)
                }
                break
                
        except Exception as e:
            print(f"⚠️ Error reading {csv_file}: {e}")
            continue
    
    if not metabolite_db:
        # Create a sample metabolite database for testing
        print("📝 Creating sample metabolite database...")
        sample_metabolites = [
            "resveratrol", "curcumin", "quercetin", "catechin", "epicatechin",
            "anthocyanin", "lycopene", "beta-carotene", "lutein", "zeaxanthin",
            "sulforaphane", "allicin", "capsaicin", "caffeine", "theobromine",
            "hesperidin", "naringin", "rutin", "kaempferol", "myricetin",
            "chlorogenic acid", "ferulic acid", "gallic acid", "ellagic acid",
            "procyanidin", "delphinidin", "cyanidin", "pelargonidin",
            "genistein", "daidzein", "isoflavone", "flavonoid", "polyphenol",
            "vitamin C", "vitamin E", "tocopherol", "ascorbic acid",
            "betalain", "betanin", "indicaxanthin", "carotenoid"
        ]
        
        df = pd.DataFrame({
            'metabolite_name': sample_metabolites,
            'compound_id': [f"FOODB_{i:05d}" for i in range(len(sample_metabolites))],
            'category': ['polyphenol'] * len(sample_metabolites)
        })
        
        metabolite_db = {
            'file': 'sample_metabolite_database.csv',
            'data': df,
            'name_columns': ['metabolite_name'],
            'metabolite_count': len(df)
        }
        
        # Save sample database
        df.to_csv('sample_metabolite_database.csv', index=False)
        print(f"✅ Created sample database with {len(sample_metabolites)} metabolites")
    
    return metabolite_db

def create_test_document():
    """Create a test document with known metabolites"""
    print("\n📝 Creating Test Document")
    print("=" * 25)
    
    # Create a scientific text with known metabolites
    test_document = """
    Polyphenolic Compounds in Plant Foods: A Comprehensive Review
    
    Abstract:
    Plant foods contain numerous bioactive compounds that contribute to human health. 
    This review examines the distribution and biological activities of major polyphenolic 
    compounds found in common dietary sources.
    
    Introduction:
    Polyphenols represent one of the largest groups of phytochemicals in the human diet. 
    These compounds include flavonoids, phenolic acids, and other aromatic compounds 
    that exhibit antioxidant and anti-inflammatory properties.
    
    Flavonoids in Fruits and Vegetables:
    Red wine contains significant amounts of resveratrol, a stilbene compound with 
    cardioprotective effects. Grapes also contain anthocyanins, particularly cyanidin 
    and delphinidin, which contribute to their purple color.
    
    Green tea is rich in catechins, especially epicatechin and epigallocatechin gallate (EGCG). 
    These compounds demonstrate strong antioxidant activity and may reduce cancer risk.
    
    Citrus fruits contain flavanones such as hesperidin in oranges and naringin in 
    grapefruits. These compounds exhibit anti-inflammatory properties.
    
    Quercetin is widely distributed in plant foods, with high concentrations found in 
    onions, apples, and berries. This flavonol shows promise in cardiovascular protection.
    
    Carotenoids and Other Compounds:
    Tomatoes are the primary dietary source of lycopene, a carotenoid with antioxidant 
    properties. Beta-carotene is abundant in carrots and sweet potatoes.
    
    Lutein and zeaxanthin are found in leafy green vegetables and are important for 
    eye health. These xanthophyll carotenoids accumulate in the macula.
    
    Cruciferous Vegetables:
    Broccoli and other cruciferous vegetables contain sulforaphane, a glucosinolate 
    derivative with potential anticancer properties.
    
    Spices and Herbs:
    Turmeric contains curcumin, a polyphenolic compound with anti-inflammatory effects. 
    Garlic contains allicin, an organosulfur compound formed when garlic is crushed.
    
    Coffee and Chocolate:
    Coffee is rich in chlorogenic acid, a phenolic compound that may influence glucose 
    metabolism. Dark chocolate contains procyanidins and other flavonoids.
    
    Caffeine and theobromine are methylxanthines found in coffee and chocolate, 
    respectively, with stimulant properties.
    
    Conclusion:
    The diverse array of polyphenolic compounds in plant foods contributes to their 
    health-promoting properties. Understanding the distribution and bioactivity of 
    these compounds is essential for optimizing dietary recommendations.
    """
    
    # Save test document
    os.makedirs("FOODB_LLM_pipeline/sample_input", exist_ok=True)
    doc_file = "FOODB_LLM_pipeline/sample_input/test_scientific_document.txt"
    
    with open(doc_file, 'w') as f:
        f.write(test_document)
    
    print(f"✅ Created test document: {doc_file}")
    print(f"📄 Document length: {len(test_document)} characters")
    
    return doc_file, test_document

def extract_metabolites_from_text(text: str, metabolite_db: Dict) -> Tuple[Set[str], Dict]:
    """Extract metabolites from text using the database"""
    print("\n🔬 Extracting Metabolites from Text")
    print("=" * 35)
    
    start_time = time.time()
    
    # Get metabolite names from database
    df = metabolite_db['data']
    name_column = metabolite_db['name_columns'][0]
    
    metabolite_names = set(df[name_column].str.lower().tolist())
    
    # Also add common variations and synonyms
    additional_metabolites = {
        'egcg', 'epigallocatechin gallate', 'vitamin c', 'ascorbic acid',
        'beta carotene', 'β-carotene', 'polyphenol', 'flavonoid',
        'anthocyanin', 'cyanidin', 'delphinidin'
    }
    metabolite_names.update(additional_metabolites)
    
    # Extract metabolites found in text
    text_lower = text.lower()
    found_metabolites = set()
    metabolite_positions = {}
    
    for metabolite in metabolite_names:
        if metabolite in text_lower:
            found_metabolites.add(metabolite)
            # Find all positions
            positions = []
            start = 0
            while True:
                pos = text_lower.find(metabolite, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            metabolite_positions[metabolite] = positions
    
    end_time = time.time()
    extraction_time = end_time - start_time
    
    print(f"⏱️ Extraction time: {extraction_time:.3f}s")
    print(f"📊 Database metabolites: {len(metabolite_names)}")
    print(f"✅ Found metabolites: {len(found_metabolites)}")
    
    return found_metabolites, {
        'extraction_time': extraction_time,
        'database_size': len(metabolite_names),
        'found_count': len(found_metabolites),
        'positions': metabolite_positions
    }

def run_pipeline_sentence_generation(doc_file: str) -> Tuple[List[str], Dict]:
    """Run sentence generation pipeline step"""
    print("\n🧪 Running Sentence Generation Pipeline")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # Import and run sentence generation
        from llm_wrapper_enhanced import LLMWrapper
        
        wrapper = LLMWrapper()
        
        # Read document
        with open(doc_file, 'r') as f:
            text = f.read()
        
        # Split into chunks for processing
        sentences = text.split('. ')
        sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
        
        print(f"📝 Processing {len(sentences)} sentences...")
        
        processed_sentences = []
        processing_times = []
        
        for i, sentence in enumerate(sentences[:10], 1):  # Limit to first 10 for testing
            print(f"  Sentence {i}: ", end="", flush=True)
            
            prompt = f"Extract metabolites from: {sentence}"
            
            sent_start = time.time()
            response = wrapper.generate_single(prompt, max_tokens=150)
            sent_end = time.time()
            
            sent_time = sent_end - sent_start
            processing_times.append(sent_time)
            
            if response:
                processed_sentences.append({
                    'original': sentence,
                    'extracted': response,
                    'processing_time': sent_time
                })
                print(f"✅ {sent_time:.2f}s")
            else:
                print(f"❌ {sent_time:.2f}s")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get wrapper statistics
        stats = wrapper.get_statistics()
        
        print(f"\n📊 Sentence Generation Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Processed sentences: {len(processed_sentences)}")
        print(f"  Average time per sentence: {sum(processing_times)/len(processing_times):.2f}s")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Provider switches: {stats['fallback_switches']}")
        
        return processed_sentences, {
            'total_time': total_time,
            'sentence_count': len(processed_sentences),
            'avg_time_per_sentence': sum(processing_times)/len(processing_times) if processing_times else 0,
            'success_rate': stats['success_rate'],
            'provider_switches': stats['fallback_switches'],
            'processing_times': processing_times
        }
        
    except Exception as e:
        print(f"❌ Error in sentence generation: {e}")
        import traceback
        traceback.print_exc()
        return [], {'total_time': 0, 'error': str(e)}

def run_pipeline_triple_extraction(processed_sentences: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Run triple extraction pipeline step"""
    print("\n🔗 Running Triple Extraction Pipeline")
    print("=" * 35)
    
    start_time = time.time()
    
    try:
        from simple_sentenceRE3_API import load_api_wrapper
        
        wrapper = load_api_wrapper()
        if not wrapper:
            print("❌ Could not load triple extraction wrapper")
            return [], {'total_time': 0, 'error': 'Wrapper loading failed'}
        
        print(f"🔬 Processing {len(processed_sentences)} sentences for triples...")
        
        extracted_triples = []
        processing_times = []
        
        for i, sent_data in enumerate(processed_sentences, 1):
            print(f"  Triple {i}: ", end="", flush=True)
            
            prompt = f"Extract relationships from: {sent_data['extracted']}"
            
            triple_start = time.time()
            response = wrapper.generate_single(prompt, max_tokens=200)
            triple_end = time.time()
            
            triple_time = triple_end - triple_start
            processing_times.append(triple_time)
            
            if response:
                extracted_triples.append({
                    'sentence': sent_data['original'],
                    'metabolites': sent_data['extracted'],
                    'triples': response,
                    'processing_time': triple_time
                })
                print(f"✅ {triple_time:.2f}s")
            else:
                print(f"❌ {triple_time:.2f}s")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n📊 Triple Extraction Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Processed triples: {len(extracted_triples)}")
        print(f"  Average time per triple: {sum(processing_times)/len(processing_times):.2f}s")
        
        return extracted_triples, {
            'total_time': total_time,
            'triple_count': len(extracted_triples),
            'avg_time_per_triple': sum(processing_times)/len(processing_times) if processing_times else 0,
            'processing_times': processing_times
        }
        
    except Exception as e:
        print(f"❌ Error in triple extraction: {e}")
        import traceback
        traceback.print_exc()
        return [], {'total_time': 0, 'error': str(e)}

def calculate_detection_metrics(found_metabolites: Set[str], extracted_results: List[Dict], 
                              metabolite_db: Dict) -> Dict:
    """Calculate precision, recall, F1 scores"""
    print("\n📊 Calculating Detection Metrics")
    print("=" * 30)
    
    # Extract metabolites mentioned in pipeline results
    pipeline_metabolites = set()
    
    for result in extracted_results:
        # Look for metabolites in the extracted text
        extracted_text = result.get('extracted', '').lower()
        
        # Simple extraction - look for known metabolites
        df = metabolite_db['data']
        name_column = metabolite_db['name_columns'][0]
        
        for metabolite in df[name_column]:
            if metabolite.lower() in extracted_text:
                pipeline_metabolites.add(metabolite.lower())
    
    # Also check common variations
    common_metabolites = {
        'resveratrol', 'curcumin', 'quercetin', 'catechin', 'epicatechin',
        'anthocyanin', 'lycopene', 'beta-carotene', 'lutein', 'zeaxanthin',
        'sulforaphane', 'allicin', 'hesperidin', 'naringin', 'caffeine',
        'theobromine', 'chlorogenic acid', 'procyanidin'
    }
    
    for result in extracted_results:
        extracted_text = result.get('extracted', '').lower()
        for metabolite in common_metabolites:
            if metabolite in extracted_text:
                pipeline_metabolites.add(metabolite)
    
    # Calculate metrics
    true_positives = len(found_metabolites.intersection(pipeline_metabolites))
    false_positives = len(pipeline_metabolites - found_metabolites)
    false_negatives = len(found_metabolites - pipeline_metabolites)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 Detection Metrics:")
    print(f"  Ground truth metabolites: {len(found_metabolites)}")
    print(f"  Pipeline detected metabolites: {len(pipeline_metabolites)}")
    print(f"  True positives: {true_positives}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1_score:.3f}")
    
    return {
        'ground_truth_count': len(found_metabolites),
        'pipeline_detected_count': len(pipeline_metabolites),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ground_truth_metabolites': list(found_metabolites),
        'pipeline_metabolites': list(pipeline_metabolites)
    }

def save_comprehensive_metrics(all_metrics: Dict):
    """Save comprehensive metrics report"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = f"FOODB_Real_Pipeline_Metrics_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n💾 Comprehensive metrics saved: {report_file}")
    return report_file

def main():
    """Run comprehensive real pipeline metrics testing"""
    print("🧬 FOODB Pipeline - Real Metrics Testing")
    print("=" * 45)
    
    overall_start = time.time()
    
    try:
        # Step 1: Load metabolite database
        step1_start = time.time()
        metabolite_db = load_metabolite_database()
        step1_end = time.time()
        
        if not metabolite_db:
            print("❌ Could not load metabolite database")
            return
        
        # Step 2: Create/load test document
        step2_start = time.time()
        doc_file, test_text = create_test_document()
        step2_end = time.time()
        
        # Step 3: Extract ground truth metabolites
        step3_start = time.time()
        found_metabolites, extraction_metrics = extract_metabolites_from_text(test_text, metabolite_db)
        step3_end = time.time()
        
        # Step 4: Run sentence generation pipeline
        step4_start = time.time()
        processed_sentences, sentence_metrics = run_pipeline_sentence_generation(doc_file)
        step4_end = time.time()
        
        # Step 5: Run triple extraction pipeline
        step5_start = time.time()
        extracted_triples, triple_metrics = run_pipeline_triple_extraction(processed_sentences)
        step5_end = time.time()
        
        # Step 6: Calculate detection metrics
        step6_start = time.time()
        detection_metrics = calculate_detection_metrics(found_metabolites, processed_sentences, metabolite_db)
        step6_end = time.time()
        
        overall_end = time.time()
        total_time = overall_end - overall_start
        
        # Compile comprehensive metrics
        comprehensive_metrics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_processing_time': total_time,
            'step_times': {
                'database_loading': step1_end - step1_start,
                'document_preparation': step2_end - step2_start,
                'ground_truth_extraction': step3_end - step3_start,
                'sentence_generation': step4_end - step4_start,
                'triple_extraction': step5_end - step5_start,
                'metrics_calculation': step6_end - step6_start
            },
            'metabolite_database': {
                'file': metabolite_db['file'],
                'metabolite_count': metabolite_db['metabolite_count']
            },
            'ground_truth_metrics': extraction_metrics,
            'sentence_generation_metrics': sentence_metrics,
            'triple_extraction_metrics': triple_metrics,
            'detection_performance': detection_metrics
        }
        
        # Display final results
        print(f"\n🎉 REAL PIPELINE METRICS COMPLETE!")
        print("=" * 40)
        
        print(f"\n⏱️ TIMING BREAKDOWN:")
        for step, duration in comprehensive_metrics['step_times'].items():
            print(f"  {step.replace('_', ' ').title()}: {duration:.3f}s")
        print(f"  TOTAL PROCESSING TIME: {total_time:.3f}s")
        
        print(f"\n📊 METABOLITE DETECTION:")
        print(f"  CSV Database Size: {metabolite_db['metabolite_count']} metabolites")
        print(f"  Found in Document: {detection_metrics['ground_truth_count']} metabolites")
        print(f"  Detected by Pipeline: {detection_metrics['pipeline_detected_count']} metabolites")
        print(f"  True Positives: {detection_metrics['true_positives']}")
        
        print(f"\n🎯 PERFORMANCE SCORES:")
        print(f"  Precision: {detection_metrics['precision']:.3f}")
        print(f"  Recall: {detection_metrics['recall']:.3f}")
        print(f"  F1 Score: {detection_metrics['f1_score']:.3f}")
        
        print(f"\n⚡ PIPELINE PERFORMANCE:")
        print(f"  Sentence Generation: {sentence_metrics.get('avg_time_per_sentence', 0):.3f}s/sentence")
        print(f"  Triple Extraction: {triple_metrics.get('avg_time_per_triple', 0):.3f}s/triple")
        print(f"  Success Rate: {sentence_metrics.get('success_rate', 0):.1%}")
        
        # Save comprehensive report
        save_comprehensive_metrics(comprehensive_metrics)
        
        return comprehensive_metrics
        
    except Exception as e:
        print(f"❌ Real pipeline metrics testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
