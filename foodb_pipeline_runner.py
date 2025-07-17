#!/usr/bin/env python3
"""
FOODB Pipeline Runner
Implements the actual pipeline execution with command-line arguments
"""

import os
import sys
import time
import json
import pandas as pd
import PyPDF2
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add FOODB pipeline to path
sys.path.append('FOODB_LLM_pipeline')

def run_pipeline(args, logger):
    """Run the complete FOODB pipeline with given arguments"""
    logger.info("🚀 Starting FOODB pipeline execution")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Initialize progress tracking
    total_files = len(args.input_files)
    results = []
    
    # Process each input file
    for i, input_file in enumerate(args.input_files, 1):
        if not args.quiet:
            print(f"\n📄 Processing file {i}/{total_files}: {input_file}")
        
        logger.info(f"Processing file {i}/{total_files}: {input_file}")
        
        # Check if should skip existing
        if args.skip_existing:
            output_file = get_output_filename(input_file, args)
            if output_file.exists():
                logger.info(f"⏭️ Skipping existing result: {output_file}")
                continue
        
        try:
            result = process_single_file(input_file, args, logger)
            results.append(result)
            
            # Save individual result
            save_results(result, input_file, args, logger)
            
            if not args.quiet:
                print(f"✅ Successfully processed {input_file}")
            logger.info(f"✅ Successfully processed {input_file}")
            
        except Exception as e:
            error_msg = f"❌ Failed to process {input_file}: {e}"
            print(error_msg)
            logger.error(error_msg)
            
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Generate batch summary if processing multiple files
    if args.batch_mode and len(results) > 1:
        generate_batch_summary(results, args, logger)
    
    logger.info(f"🎉 Pipeline execution completed. Processed {len(results)}/{total_files} files successfully.")

def process_single_file(input_file: str, args, logger) -> Dict[str, Any]:
    """Process a single PDF file through the pipeline"""
    start_time = time.time()
    
    # Step 1: Extract text from PDF
    if args.verbose:
        print("   📄 Step 1: Extracting text from PDF...")
    logger.info("Step 1: Extracting text from PDF")
    pdf_result = extract_pdf_text(input_file, logger)
    
    # Step 2: Chunk text
    if args.verbose:
        print(f"   📝 Step 2: Chunking text into {args.chunk_size}-character chunks...")
    logger.info("Step 2: Chunking text")
    chunks = chunk_text(pdf_result['text'], args, logger)
    
    # Step 3: Extract metabolites
    if args.verbose:
        print(f"   🧬 Step 3: Extracting metabolites from {len(chunks)} chunks...")
    logger.info("Step 3: Extracting metabolites")
    extraction_result = extract_metabolites(chunks, args, logger)
    
    # Step 4: Match against database
    if args.verbose:
        print(f"   📊 Step 4: Matching against database...")
    logger.info("Step 4: Matching against database")
    matching_result = match_database(extraction_result['metabolites'], args, logger)
    
    # Step 5: Calculate metrics
    if args.calculate_metrics:
        if args.verbose:
            print(f"   📈 Step 5: Calculating performance metrics...")
        logger.info("Step 5: Calculating metrics")
        metrics = calculate_metrics(matching_result, args, logger)
    else:
        metrics = {}
    
    end_time = time.time()
    
    return {
        'input_file': input_file,
        'processing_time': end_time - start_time,
        'pdf_result': pdf_result,
        'chunks': chunks if args.save_chunks else {'count': len(chunks)},
        'extraction_result': extraction_result,
        'matching_result': matching_result,
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'chunk_size': args.chunk_size,
            'max_tokens': args.max_tokens,
            'document_only': args.document_only,
            'verify_compounds': args.verify_compounds,
            'providers': args.providers
        }
    }

def extract_pdf_text(input_file: str, logger) -> Dict[str, Any]:
    """Extract text from PDF file"""
    start_time = time.time()
    
    try:
        with open(input_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        end_time = time.time()
        
        result = {
            'text': text,
            'pages': len(pdf_reader.pages),
            'characters': len(text),
            'extraction_time': end_time - start_time
        }
        
        logger.info(f"📄 Extracted {len(text):,} characters from {len(pdf_reader.pages)} pages in {result['extraction_time']:.3f}s")
        return result
        
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        raise

def chunk_text(text: str, args, logger) -> List[str]:
    """Chunk text according to configuration"""
    start_time = time.time()
    
    chunks = []
    chunk_size = args.chunk_size
    overlap = args.chunk_overlap
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        
        # Skip chunks that are too small
        if len(chunk) >= args.min_chunk_size:
            chunks.append(chunk)
    
    end_time = time.time()
    
    logger.info(f"📝 Created {len(chunks)} chunks in {end_time - start_time:.3f}s")
    return chunks

def extract_metabolites(chunks: List[str], args, logger) -> Dict[str, Any]:
    """Extract metabolites from text chunks"""
    start_time = time.time()
    
    try:
        from llm_wrapper_enhanced import LLMWrapper, RetryConfig
        
        # Configure retry settings
        retry_config = RetryConfig(
            max_attempts=args.max_attempts,
            base_delay=args.base_delay,
            max_delay=args.max_delay,
            exponential_base=args.exponential_base,
            jitter=not args.disable_jitter
        )
        
        # Initialize wrapper
        wrapper = LLMWrapper(
            retry_config=retry_config,
            document_only_mode=args.document_only
        )
        
        # Set provider order
        wrapper.fallback_order = args.providers
        if args.primary_provider:
            wrapper.current_provider = args.primary_provider
        
        all_metabolites = []
        chunk_results = []
        successful_chunks = 0
        failed_chunks = 0
        
        # Process chunks
        for i, chunk in enumerate(chunks, 1):
            if args.resume_from_chunk and i < args.resume_from_chunk:
                continue
            
            if args.verbose:
                print(f"      Chunk {i:2d}/{len(chunks)}: ", end="", flush=True)
            
            chunk_start = time.time()
            
            try:
                # Extract metabolites
                if args.document_only:
                    response = wrapper.extract_metabolites_document_only(chunk, args.max_tokens)
                else:
                    prompt = args.custom_prompt or f"Extract metabolites and biomarkers from this text:\n\n{chunk}"
                    response = wrapper.generate_single_with_fallback(prompt, args.max_tokens)
                
                # Parse response
                metabolites = parse_metabolite_response(response)
                
                # Verify compounds if requested
                if args.verify_compounds and metabolites:
                    verification = wrapper.verify_compounds_in_text(chunk, metabolites, args.max_tokens)
                    verified_metabolites = parse_verification_response(verification, metabolites)
                    metabolites = verified_metabolites
                
                all_metabolites.extend(metabolites)
                
                chunk_end = time.time()
                chunk_time = chunk_end - chunk_start
                
                chunk_results.append({
                    'chunk_id': i,
                    'processing_time': chunk_time,
                    'metabolites_found': len(metabolites),
                    'success': True,
                    'metabolites': metabolites if args.save_raw_responses else None,
                    'raw_response': response if args.save_raw_responses else None
                })
                
                successful_chunks += 1
                
                if args.verbose:
                    print(f"✅ {chunk_time:.2f}s ({len(metabolites)} compounds)")
                
            except Exception as e:
                chunk_end = time.time()
                chunk_time = chunk_end - chunk_start
                
                chunk_results.append({
                    'chunk_id': i,
                    'processing_time': chunk_time,
                    'metabolites_found': 0,
                    'success': False,
                    'error': str(e)
                })
                
                failed_chunks += 1
                
                if args.verbose:
                    print(f"❌ {chunk_time:.2f}s (failed)")
                
                logger.warning(f"Chunk {i} failed: {e}")
        
        end_time = time.time()
        
        # Get final statistics
        stats = wrapper.get_statistics()
        
        result = {
            'metabolites': list(set(all_metabolites)),  # Remove duplicates
            'total_metabolites': len(all_metabolites),
            'unique_metabolites': len(set(all_metabolites)),
            'successful_chunks': successful_chunks,
            'failed_chunks': failed_chunks,
            'chunk_results': chunk_results,
            'processing_time': end_time - start_time,
            'llm_statistics': stats
        }
        
        logger.info(f"🧬 Extracted {result['unique_metabolites']} unique metabolites from {successful_chunks}/{len(chunks)} chunks in {result['processing_time']:.3f}s")
        return result
        
    except Exception as e:
        logger.error(f"❌ Metabolite extraction failed: {e}")
        raise

def parse_metabolite_response(response: str) -> List[str]:
    """Parse metabolites from LLM response"""
    if not response or response.lower().strip() in ['no compounds found', 'no specific compounds mentioned']:
        return []
    
    metabolites = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '•')):
            line = line[2:].strip()
        
        # Skip common non-compound responses
        if any(skip in line.lower() for skip in [
            'based on', 'however', 'note that', 'unfortunately', 'please note',
            'i can provide', 'the text', 'no specific', 'not mentioned'
        ]):
            continue
        
        if line and len(line) > 2:
            metabolites.append(line)
    
    return metabolites

def parse_verification_response(verification: str, original_metabolites: List[str]) -> List[str]:
    """Parse verification response to get verified metabolites"""
    verified = []
    
    for metabolite in original_metabolites:
        # Simple verification parsing - look for FOUND
        if f"{metabolite}: FOUND" in verification or f"{metabolite.lower()}: found" in verification.lower():
            verified.append(metabolite)
    
    return verified

def match_database(metabolites: List[str], args, logger) -> Dict[str, Any]:
    """Match extracted metabolites against CSV database"""
    start_time = time.time()

    try:
        # Load CSV database
        df = pd.read_csv(args.csv_database)
        csv_biomarkers = set(df[args.csv_column].str.lower().str.strip().tolist())

        # Match metabolites
        matches = set()
        metabolites_set = set(m.lower().strip() for m in metabolites)

        for metabolite in metabolites_set:
            for csv_biomarker in csv_biomarkers:
                # Exact match
                if metabolite == csv_biomarker:
                    matches.add(csv_biomarker)
                    break
                # Meaningful partial match
                elif len(metabolite) > 5 and metabolite in csv_biomarker:
                    matches.add(csv_biomarker)
                    break
                elif len(csv_biomarker) > 5 and csv_biomarker in metabolite:
                    matches.add(csv_biomarker)
                    break

        end_time = time.time()

        result = {
            'csv_database_size': len(csv_biomarkers),
            'extracted_metabolites': len(metabolites_set),
            'matches_found': len(matches),
            'detection_rate': len(matches) / len(csv_biomarkers) if csv_biomarkers else 0,
            'matched_biomarkers': sorted(list(matches)),
            'non_csv_metabolites': sorted(list(metabolites_set - {m for m in metabolites_set if any(csv in m or m in csv for csv in csv_biomarkers)})),
            'processing_time': end_time - start_time
        }

        logger.info(f"📊 Found {len(matches)}/{len(csv_biomarkers)} biomarkers ({result['detection_rate']:.1%}) in {result['processing_time']:.3f}s")
        return result

    except Exception as e:
        logger.error(f"❌ Database matching failed: {e}")
        raise

def calculate_metrics(matching_result: Dict[str, Any], args, logger) -> Dict[str, Any]:
    """Calculate performance metrics"""
    start_time = time.time()

    try:
        true_positives = matching_result['matches_found']
        false_positives = matching_result['extracted_metabolites'] - true_positives
        false_negatives = matching_result['csv_database_size'] - true_positives

        precision = true_positives / matching_result['extracted_metabolites'] if matching_result['extracted_metabolites'] > 0 else 0
        recall = true_positives / matching_result['csv_database_size'] if matching_result['csv_database_size'] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        end_time = time.time()

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'accuracy': recall,  # For this task, accuracy = recall
            'processing_time': end_time - start_time
        }

        logger.info(f"📈 Metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")
        return metrics

    except Exception as e:
        logger.error(f"❌ Metrics calculation failed: {e}")
        raise

def get_output_filename(input_file: str, args) -> Path:
    """Generate output filename based on input file and arguments"""
    input_path = Path(input_file)
    output_dir = Path(args.output_dir)

    # Create base filename
    base_name = input_path.stem
    if args.output_prefix:
        base_name = f"{args.output_prefix}_{base_name}"

    # Add timestamp if batch mode
    if args.batch_mode:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f"{base_name}_{timestamp}"

    return output_dir / f"{base_name}_results.json"

def save_results(result: Dict[str, Any], input_file: str, args, logger):
    """Save results to output files"""
    try:
        output_file = get_output_filename(input_file, args)

        # Save JSON results
        if args.export_format in ['json', 'all']:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"💾 Saved JSON results: {output_file}")

        # Save CSV results
        if args.export_format in ['csv', 'all']:
            csv_file = output_file.with_suffix('.csv')
            save_csv_results(result, csv_file, logger)

        # Save Excel results
        if args.export_format in ['xlsx', 'all']:
            xlsx_file = output_file.with_suffix('.xlsx')
            save_xlsx_results(result, xlsx_file, logger)

        # Save timing analysis if requested
        if args.save_timing:
            timing_file = output_file.with_name(f"{output_file.stem}_timing.json")
            save_timing_analysis(result, timing_file, logger)

        # Save chunks if requested
        if args.save_chunks and isinstance(result['chunks'], list):
            chunks_dir = output_file.parent / f"{output_file.stem}_chunks"
            save_chunks(result['chunks'], chunks_dir, logger)

    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise

def save_csv_results(result: Dict[str, Any], csv_file: Path, logger):
    """Save results in CSV format"""
    try:
        # Create DataFrame with key results
        data = {
            'Metabolite': result['extraction_result']['metabolites'],
            'In_Database': [m in result['matching_result']['matched_biomarkers'] for m in result['extraction_result']['metabolites']]
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 Saved CSV results: {csv_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save CSV: {e}")

def save_xlsx_results(result: Dict[str, Any], xlsx_file: Path, logger):
    """Save results in Excel format"""
    try:
        with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
            # Metabolites sheet
            metabolites_df = pd.DataFrame({
                'Metabolite': result['extraction_result']['metabolites'],
                'In_Database': [m in result['matching_result']['matched_biomarkers'] for m in result['extraction_result']['metabolites']]
            })
            metabolites_df.to_excel(writer, sheet_name='Metabolites', index=False)

            # Metrics sheet
            if result['metrics']:
                metrics_df = pd.DataFrame([result['metrics']])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

            # Summary sheet
            summary_data = {
                'Metric': ['Total Metabolites', 'Unique Metabolites', 'Database Matches', 'Detection Rate', 'Processing Time'],
                'Value': [
                    result['extraction_result']['total_metabolites'],
                    result['extraction_result']['unique_metabolites'],
                    result['matching_result']['matches_found'],
                    f"{result['matching_result']['detection_rate']:.1%}",
                    f"{result['processing_time']:.2f}s"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        logger.info(f"💾 Saved Excel results: {xlsx_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save Excel: {e}")

def save_timing_analysis(result: Dict[str, Any], timing_file: Path, logger):
    """Save detailed timing analysis"""
    try:
        timing_data = {
            'total_processing_time': result['processing_time'],
            'pdf_extraction_time': result['pdf_result']['extraction_time'],
            'metabolite_extraction_time': result['extraction_result']['processing_time'],
            'database_matching_time': result['matching_result']['processing_time'],
            'chunk_timings': result['extraction_result']['chunk_results']
        }

        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)

        logger.info(f"💾 Saved timing analysis: {timing_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save timing analysis: {e}")

def save_chunks(chunks: List[str], chunks_dir: Path, logger):
    """Save text chunks to separate files"""
    try:
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks, 1):
            chunk_file = chunks_dir / f"chunk_{i:03d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)

        logger.info(f"💾 Saved {len(chunks)} chunks to: {chunks_dir}")

    except Exception as e:
        logger.error(f"❌ Failed to save chunks: {e}")

def generate_batch_summary(results: List[Dict[str, Any]], args, logger):
    """Generate summary for batch processing"""
    try:
        summary_file = Path(args.output_dir) / "batch_summary.json"

        # Calculate batch statistics
        total_files = len(results)
        total_metabolites = sum(r['extraction_result']['unique_metabolites'] for r in results)
        total_matches = sum(r['matching_result']['matches_found'] for r in results)
        total_time = sum(r['processing_time'] for r in results)
        avg_detection_rate = sum(r['matching_result']['detection_rate'] for r in results) / total_files

        batch_summary = {
            'batch_statistics': {
                'total_files_processed': total_files,
                'total_metabolites_extracted': total_metabolites,
                'total_database_matches': total_matches,
                'average_detection_rate': avg_detection_rate,
                'total_processing_time': total_time,
                'average_time_per_file': total_time / total_files
            },
            'file_results': [
                {
                    'file': r['input_file'],
                    'metabolites': r['extraction_result']['unique_metabolites'],
                    'matches': r['matching_result']['matches_found'],
                    'detection_rate': r['matching_result']['detection_rate'],
                    'processing_time': r['processing_time']
                }
                for r in results
            ],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)

        logger.info(f"📊 Generated batch summary: {summary_file}")

        # Print summary to console
        if not args.quiet:
            print(f"\n📊 BATCH PROCESSING SUMMARY")
            print(f"=" * 30)
            print(f"Files processed: {total_files}")
            print(f"Total metabolites: {total_metabolites}")
            print(f"Total matches: {total_matches}")
            print(f"Average detection rate: {avg_detection_rate:.1%}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per file: {total_time/total_files:.2f}s")

    except Exception as e:
        logger.error(f"❌ Failed to generate batch summary: {e}")
