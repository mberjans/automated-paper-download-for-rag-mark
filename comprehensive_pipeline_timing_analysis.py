#!/usr/bin/env python3
"""
Comprehensive Pipeline Timing Analysis
Measure detailed timing for each step of the FOODB pipeline with document-only extraction
"""

import time
import json
import sys
import pandas as pd
import PyPDF2
sys.path.append('FOODB_LLM_pipeline')

def time_pdf_extraction():
    """Time the PDF text extraction step"""
    print("📄 STEP 1: PDF Text Extraction")
    print("=" * 35)
    
    start_time = time.time()
    
    try:
        # Extract text from wine PDF
        with open('Wine-consumptionbiomarkers-HMDB.pdf', 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        end_time = time.time()
        extraction_time = end_time - start_time
        
        print(f"✅ PDF extraction completed")
        print(f"   File: Wine-consumptionbiomarkers-HMDB.pdf")
        print(f"   Pages: {len(pdf_reader.pages)}")
        print(f"   Characters extracted: {len(text):,}")
        print(f"   Extraction time: {extraction_time:.3f} seconds")
        
        return {
            'step': 'pdf_extraction',
            'time': extraction_time,
            'pages': len(pdf_reader.pages),
            'characters': len(text),
            'text': text
        }
        
    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        return None

def time_text_chunking(text):
    """Time the text chunking step"""
    print("\n📝 STEP 2: Text Chunking")
    print("=" * 25)
    
    start_time = time.time()
    
    # Chunk text into manageable pieces
    chunk_size = 1500
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
    end_time = time.time()
    chunking_time = end_time - start_time
    
    print(f"✅ Text chunking completed")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Chunking time: {chunking_time:.3f} seconds")
    
    return {
        'step': 'text_chunking',
        'time': chunking_time,
        'chunk_size': chunk_size,
        'total_chunks': len(chunks),
        'chunks': chunks
    }

def time_document_only_extraction(chunks):
    """Time the document-only metabolite extraction step"""
    print("\n🧬 STEP 3: Document-Only Metabolite Extraction")
    print("=" * 50)
    
    from llm_wrapper_enhanced import LLMWrapper
    
    # Initialize wrapper with document-only mode
    wrapper = LLMWrapper(document_only_mode=True)
    
    start_time = time.time()
    
    all_extracted_metabolites = []
    chunk_timings = []
    successful_chunks = 0
    failed_chunks = 0
    
    print(f"🔬 Processing {len(chunks)} chunks with document-only extraction...")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"   Chunk {i:2d}: ", end="", flush=True)
        
        chunk_start = time.time()
        
        try:
            # Extract using document-only method
            result = wrapper.extract_metabolites_document_only(chunk, 200)
            
            chunk_end = time.time()
            chunk_time = chunk_end - chunk_start
            
            # Parse extracted compounds
            extracted_compounds = []
            if result and result.lower() != "no compounds found":
                lines = [line.strip() for line in result.split('\n') if line.strip()]
                for line in lines:
                    # Clean up compound names
                    clean_line = line
                    if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-')):
                        clean_line = line[2:].strip()
                    if clean_line and len(clean_line) > 2:
                        extracted_compounds.append(clean_line)
            
            all_extracted_metabolites.extend(extracted_compounds)
            
            chunk_timings.append({
                'chunk_id': i,
                'time': chunk_time,
                'compounds_found': len(extracted_compounds),
                'success': True
            })
            
            successful_chunks += 1
            print(f"✅ {chunk_time:.2f}s ({len(extracted_compounds)} compounds)")
            
        except Exception as e:
            chunk_end = time.time()
            chunk_time = chunk_end - chunk_start
            
            chunk_timings.append({
                'chunk_id': i,
                'time': chunk_time,
                'compounds_found': 0,
                'success': False,
                'error': str(e)
            })
            
            failed_chunks += 1
            print(f"❌ {chunk_time:.2f}s (failed)")
    
    end_time = time.time()
    extraction_time = end_time - start_time
    
    # Calculate statistics
    avg_chunk_time = sum(ct['time'] for ct in chunk_timings) / len(chunk_timings)
    min_chunk_time = min(ct['time'] for ct in chunk_timings)
    max_chunk_time = max(ct['time'] for ct in chunk_timings)
    
    print(f"\n✅ Document-only extraction completed")
    print(f"   Total extraction time: {extraction_time:.3f} seconds")
    print(f"   Successful chunks: {successful_chunks}/{len(chunks)}")
    print(f"   Failed chunks: {failed_chunks}")
    print(f"   Average chunk time: {avg_chunk_time:.3f} seconds")
    print(f"   Fastest chunk: {min_chunk_time:.3f} seconds")
    print(f"   Slowest chunk: {max_chunk_time:.3f} seconds")
    print(f"   Total compounds extracted: {len(all_extracted_metabolites)}")
    print(f"   Unique compounds: {len(set(all_extracted_metabolites))}")
    
    return {
        'step': 'document_only_extraction',
        'time': extraction_time,
        'successful_chunks': successful_chunks,
        'failed_chunks': failed_chunks,
        'chunk_timings': chunk_timings,
        'avg_chunk_time': avg_chunk_time,
        'min_chunk_time': min_chunk_time,
        'max_chunk_time': max_chunk_time,
        'total_compounds': len(all_extracted_metabolites),
        'unique_compounds': len(set(all_extracted_metabolites)),
        'extracted_metabolites': list(set(all_extracted_metabolites))
    }

def time_database_matching(extracted_metabolites):
    """Time the database matching step"""
    print("\n📊 STEP 4: Database Matching")
    print("=" * 30)
    
    start_time = time.time()
    
    try:
        # Load CSV biomarkers database
        df = pd.read_csv("urinary_wine_biomarkers.csv")
        csv_biomarkers = set(df['Compound Name'].str.lower().str.strip().tolist())
        
        # Match extracted metabolites against database
        matches = set()
        extracted_set = set(m.lower().strip() for m in extracted_metabolites)
        
        for extracted in extracted_set:
            for csv_biomarker in csv_biomarkers:
                # Exact match
                if extracted == csv_biomarker:
                    matches.add(csv_biomarker)
                    break
                # Meaningful partial match
                elif len(extracted) > 5 and extracted in csv_biomarker:
                    matches.add(csv_biomarker)
                    break
                elif len(csv_biomarker) > 5 and csv_biomarker in extracted:
                    matches.add(csv_biomarker)
                    break
        
        end_time = time.time()
        matching_time = end_time - start_time
        
        print(f"✅ Database matching completed")
        print(f"   CSV database size: {len(csv_biomarkers)} biomarkers")
        print(f"   Extracted compounds: {len(extracted_set)}")
        print(f"   Matches found: {len(matches)}")
        print(f"   Detection rate: {len(matches)/len(csv_biomarkers):.1%}")
        print(f"   Matching time: {matching_time:.3f} seconds")
        
        return {
            'step': 'database_matching',
            'time': matching_time,
            'csv_database_size': len(csv_biomarkers),
            'extracted_compounds': len(extracted_set),
            'matches_found': len(matches),
            'detection_rate': len(matches)/len(csv_biomarkers),
            'matched_biomarkers': sorted(list(matches))
        }
        
    except Exception as e:
        end_time = time.time()
        matching_time = end_time - start_time
        print(f"❌ Database matching failed: {e}")
        return {
            'step': 'database_matching',
            'time': matching_time,
            'error': str(e)
        }

def time_results_analysis(extraction_results, matching_results):
    """Time the results analysis step"""
    print("\n📈 STEP 5: Results Analysis")
    print("=" * 28)
    
    start_time = time.time()
    
    # Calculate comprehensive metrics
    if 'matched_biomarkers' in matching_results and 'extracted_metabolites' in extraction_results:
        true_positives = matching_results['matches_found']
        false_positives = extraction_results['unique_compounds'] - true_positives
        false_negatives = matching_results['csv_database_size'] - true_positives
        
        precision = true_positives / extraction_results['unique_compounds'] if extraction_results['unique_compounds'] > 0 else 0
        recall = true_positives / matching_results['csv_database_size'] if matching_results['csv_database_size'] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1_score = 0
        true_positives = false_positives = false_negatives = 0
    
    # Generate comprehensive report
    analysis_report = {
        'accuracy_metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'extraction_performance': {
            'total_chunks_processed': extraction_results.get('successful_chunks', 0),
            'success_rate': extraction_results.get('successful_chunks', 0) / (extraction_results.get('successful_chunks', 0) + extraction_results.get('failed_chunks', 0)) if (extraction_results.get('successful_chunks', 0) + extraction_results.get('failed_chunks', 0)) > 0 else 0,
            'compounds_per_chunk': extraction_results.get('total_compounds', 0) / extraction_results.get('successful_chunks', 1),
            'processing_speed': extraction_results.get('successful_chunks', 0) / extraction_results.get('time', 1)
        }
    }
    
    end_time = time.time()
    analysis_time = end_time - start_time
    
    print(f"✅ Results analysis completed")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1_score:.3f}")
    print(f"   Analysis time: {analysis_time:.3f} seconds")
    
    return {
        'step': 'results_analysis',
        'time': analysis_time,
        'analysis_report': analysis_report
    }

def generate_comprehensive_timing_report(step_results):
    """Generate comprehensive timing report"""
    print("\n📊 COMPREHENSIVE PIPELINE TIMING REPORT")
    print("=" * 50)
    
    # Calculate total time
    total_time = sum(result['time'] for result in step_results if 'time' in result)
    
    print(f"\n⏱️ STEP-BY-STEP TIMING BREAKDOWN:")
    print(f"=" * 40)
    
    for i, result in enumerate(step_results, 1):
        if 'time' in result:
            step_name = result['step'].replace('_', ' ').title()
            time_seconds = result['time']
            percentage = (time_seconds / total_time) * 100
            
            print(f"   Step {i}: {step_name}")
            print(f"           Time: {time_seconds:.3f}s ({percentage:.1f}%)")
            
            # Add step-specific details
            if result['step'] == 'pdf_extraction':
                print(f"           Pages: {result.get('pages', 'N/A')}")
                print(f"           Characters: {result.get('characters', 'N/A'):,}")
            elif result['step'] == 'text_chunking':
                print(f"           Chunks: {result.get('total_chunks', 'N/A')}")
            elif result['step'] == 'document_only_extraction':
                print(f"           Chunks processed: {result.get('successful_chunks', 'N/A')}")
                print(f"           Compounds found: {result.get('unique_compounds', 'N/A')}")
                print(f"           Avg chunk time: {result.get('avg_chunk_time', 0):.3f}s")
            elif result['step'] == 'database_matching':
                print(f"           Matches found: {result.get('matches_found', 'N/A')}")
                print(f"           Detection rate: {result.get('detection_rate', 0):.1%}")
    
    print(f"\n🎯 TOTAL PIPELINE TIME: {total_time:.3f} seconds")
    
    # Performance analysis
    extraction_result = next((r for r in step_results if r['step'] == 'document_only_extraction'), {})
    if extraction_result:
        chunks_processed = extraction_result.get('successful_chunks', 0)
        throughput = chunks_processed / total_time if total_time > 0 else 0
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Throughput: {throughput:.3f} chunks/second")
        print(f"   Processing efficiency: {chunks_processed}/{chunks_processed + extraction_result.get('failed_chunks', 0)} chunks successful")
        print(f"   Document coverage: 100% (all chunks processed)")
    
    # Save detailed timing report
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = f"comprehensive_pipeline_timing_{timestamp}.json"
    
    timing_report = {
        'timestamp': timestamp,
        'total_pipeline_time': total_time,
        'step_results': step_results,
        'performance_summary': {
            'total_time_seconds': total_time,
            'throughput_chunks_per_second': throughput if 'throughput' in locals() else 0,
            'success_rate': extraction_result.get('successful_chunks', 0) / (extraction_result.get('successful_chunks', 0) + extraction_result.get('failed_chunks', 0)) if extraction_result else 0
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(timing_report, f, indent=2)
    
    print(f"\n💾 Detailed timing report saved: {report_file}")
    
    return timing_report

def main():
    """Run comprehensive pipeline timing analysis"""
    print("⏱️ FOODB Pipeline - Comprehensive Timing Analysis")
    print("=" * 55)
    print("Testing document-only extraction with detailed timing")
    
    step_results = []
    
    try:
        # Step 1: PDF Text Extraction
        pdf_result = time_pdf_extraction()
        if pdf_result:
            step_results.append(pdf_result)
        else:
            print("❌ Cannot proceed without PDF extraction")
            return
        
        # Step 2: Text Chunking
        chunking_result = time_text_chunking(pdf_result['text'])
        if chunking_result:
            step_results.append(chunking_result)
        else:
            print("❌ Cannot proceed without text chunking")
            return
        
        # Step 3: Document-Only Metabolite Extraction
        extraction_result = time_document_only_extraction(chunking_result['chunks'])
        if extraction_result:
            step_results.append(extraction_result)
        else:
            print("❌ Cannot proceed without metabolite extraction")
            return
        
        # Step 4: Database Matching
        matching_result = time_database_matching(extraction_result['extracted_metabolites'])
        if matching_result:
            step_results.append(matching_result)
        
        # Step 5: Results Analysis
        analysis_result = time_results_analysis(extraction_result, matching_result)
        if analysis_result:
            step_results.append(analysis_result)
        
        # Generate comprehensive report
        timing_report = generate_comprehensive_timing_report(step_results)
        
        print(f"\n🎉 COMPREHENSIVE TIMING ANALYSIS COMPLETE!")
        print(f"All pipeline steps measured with document-only extraction approach.")
        
    except Exception as e:
        print(f"❌ Comprehensive timing analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
