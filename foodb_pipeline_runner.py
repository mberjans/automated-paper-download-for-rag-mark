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

    # Expand input files to handle directories
    pdf_files = expand_input_files(args.input_files, logger)

    # Detect directory mode automatically if processing multiple files from directories
    if not getattr(args, 'directory_mode', False) and len(pdf_files) > 1:
        # Check if any input was a directory
        for input_path in args.input_files:
            if os.path.isdir(input_path):
                args.directory_mode = True
                logger.info("🗂️ Directory mode automatically enabled")
                break

    # Create output directory structure
    setup_output_directories(args, logger)

    # Initialize progress tracking
    total_files = len(pdf_files)
    results = []

    # Initialize consolidated output structures
    if getattr(args, 'directory_mode', False) and getattr(args, 'consolidated_output', True):
        consolidated_data = initialize_consolidated_data()

    # Process each PDF file
    for i, pdf_file in enumerate(pdf_files, 1):
        if not args.quiet:
            print(f"\n📄 Processing file {i}/{total_files}: {pdf_file}")

        logger.info(f"Processing file {i}/{total_files}: {pdf_file}")

        # Check if should skip existing
        if args.skip_existing:
            output_file = get_output_filename(pdf_file, args)
            if output_file.exists():
                logger.info(f"⏭️ Skipping existing result: {output_file}")
                continue

        try:
            result = process_single_file(pdf_file, args, logger)
            results.append(result)

            # Save results based on mode
            if getattr(args, 'directory_mode', False):
                # Directory mode: dual output
                save_directory_mode_results(result, pdf_file, args, logger, consolidated_data if getattr(args, 'consolidated_output', True) else None)
            else:
                # Standard mode: individual files only
                save_results(result, pdf_file, args, logger)

            if not args.quiet:
                print(f"✅ Successfully processed {pdf_file}")
            logger.info(f"✅ Successfully processed {pdf_file}")

        except Exception as e:
            error_msg = f"❌ Failed to process {pdf_file}: {e}"
            print(error_msg)
            logger.error(error_msg)

            if args.debug:
                import traceback
                traceback.print_exc()

    # Finalize consolidated output
    if getattr(args, 'directory_mode', False) and getattr(args, 'consolidated_output', True) and results:
        finalize_consolidated_output(consolidated_data, results, args, logger)

    # Generate batch summary if processing multiple files
    if (args.batch_mode or getattr(args, 'directory_mode', False)) and len(results) > 1:
        generate_batch_summary(results, args, logger)

    logger.info(f"🎉 Pipeline execution completed. Processed {len(results)}/{total_files} files successfully.")

def expand_input_files(input_paths: List[str], logger) -> List[str]:
    """Expand input paths to include all PDF files from directories"""
    pdf_files = []

    for input_path in input_paths:
        if os.path.isfile(input_path):
            if input_path.lower().endswith('.pdf'):
                pdf_files.append(input_path)
            else:
                logger.warning(f"⚠️ Skipping non-PDF file: {input_path}")
        elif os.path.isdir(input_path):
            dir_path = Path(input_path)
            dir_pdfs = list(dir_path.glob("*.pdf"))
            dir_pdfs.extend(list(dir_path.glob("**/*.pdf")))  # Include subdirectories

            if dir_pdfs:
                logger.info(f"📁 Found {len(dir_pdfs)} PDF files in directory: {input_path}")
                pdf_files.extend([str(pdf) for pdf in dir_pdfs])
            else:
                logger.warning(f"⚠️ No PDF files found in directory: {input_path}")
        else:
            logger.warning(f"⚠️ Invalid input path: {input_path}")

    # Remove duplicates and sort
    pdf_files = sorted(list(set(pdf_files)))
    logger.info(f"📊 Total PDF files to process: {len(pdf_files)}")

    return pdf_files

def setup_output_directories(args, logger):
    """Setup output directory structure for different modes"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Main output directory: {output_dir}")

    if getattr(args, 'directory_mode', False):
        # Create subdirectories for directory mode
        if getattr(args, 'individual_output', True):
            individual_dir = output_dir / getattr(args, 'individual_subdir', 'individual_papers')
            individual_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Individual papers directory: {individual_dir}")

        if getattr(args, 'consolidated_output', True):
            consolidated_dir = output_dir / getattr(args, 'consolidated_subdir', 'consolidated')
            consolidated_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Consolidated results directory: {consolidated_dir}")

def initialize_consolidated_data():
    """Initialize data structures for consolidated output"""
    return {
        'all_metabolites': set(),
        'all_matches': set(),
        'paper_results': [],
        'summary_stats': {
            'total_papers': 0,
            'total_metabolites': 0,
            'total_unique_metabolites': 0,
            'total_matches': 0,
            'average_detection_rate': 0.0
        }
    }

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
            document_only_mode=args.document_only,
            groq_model=getattr(args, 'groq_model', None)
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

def save_directory_mode_results(result: Dict[str, Any], input_file: str, args, logger, consolidated_data: Optional[Dict] = None):
    """Save results in directory mode with dual output structure"""
    try:
        # Save individual paper-specific results (with timestamps)
        if getattr(args, 'individual_output', True):
            save_individual_paper_results(result, input_file, args, logger)

        # Update consolidated data
        if consolidated_data is not None:
            update_consolidated_data(consolidated_data, result, input_file)

    except Exception as e:
        logger.error(f"❌ Failed to save directory mode results: {e}")
        raise

def save_individual_paper_results(result: Dict[str, Any], input_file: str, args, logger):
    """Save individual paper results in timestamped files"""
    try:
        # Create individual output directory
        output_dir = Path(args.output_dir) / getattr(args, 'individual_subdir', 'individual_papers')

        # Generate timestamped filename for individual paper
        input_path = Path(input_file)
        base_name = input_path.stem
        if args.output_prefix:
            base_name = f"{args.output_prefix}_{base_name}"

        # Always use timestamp for individual files
        if hasattr(args, 'custom_timestamp') and args.custom_timestamp:
            timestamp = args.custom_timestamp
        else:
            timestamp_format = getattr(args, 'timestamp_format', '%Y%m%d_%H%M%S')
            timestamp = time.strftime(timestamp_format)

        base_filename = f"{base_name}_{timestamp}"

        # Save in all requested formats
        if args.export_format in ['json', 'all']:
            json_file = output_dir / f"{base_filename}.json"
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"💾 Saved individual JSON: {json_file}")

        if args.export_format in ['csv', 'all']:
            csv_file = output_dir / f"{base_filename}.csv"
            save_csv_results(result, csv_file, logger)

        if args.export_format in ['xlsx', 'all']:
            xlsx_file = output_dir / f"{base_filename}.xlsx"
            save_xlsx_results(result, xlsx_file, logger)

        # Save timing if requested
        if args.save_timing:
            timing_file = output_dir / f"{base_filename}_timing.json"
            save_timing_analysis(result, timing_file, logger)

        # Save raw responses if requested
        if args.save_raw_responses:
            raw_file = output_dir / f"{base_filename}_raw_responses.json"
            save_raw_responses(result, raw_file, logger)

    except Exception as e:
        logger.error(f"❌ Failed to save individual paper results: {e}")
        raise

def update_consolidated_data(consolidated_data: Dict, result: Dict[str, Any], input_file: str):
    """Update consolidated data structures with new result"""
    try:
        # Extract data from result
        metabolites = set(result['extraction_result']['metabolites'])
        matches = set(result['matching_result']['matched_biomarkers'])

        # Update consolidated sets
        consolidated_data['all_metabolites'].update(metabolites)
        consolidated_data['all_matches'].update(matches)

        # Add paper-specific result
        paper_result = {
            'paper_file': input_file,
            'paper_name': Path(input_file).stem,
            'processing_time': result['processing_time'],
            'metabolites_count': len(metabolites),
            'matches_count': len(matches),
            'detection_rate': result['matching_result']['detection_rate'],
            'precision': result['metrics'].get('precision', 0) if 'metrics' in result else 0,
            'recall': result['metrics'].get('recall', 0) if 'metrics' in result else 0,
            'f1_score': result['metrics'].get('f1_score', 0) if 'metrics' in result else 0,
            'metabolites': list(metabolites),
            'matched_biomarkers': list(matches)
        }

        consolidated_data['paper_results'].append(paper_result)

        # Update summary stats
        consolidated_data['summary_stats']['total_papers'] += 1
        consolidated_data['summary_stats']['total_metabolites'] += len(metabolites)

    except Exception as e:
        print(f"❌ Error updating consolidated data: {e}")

def finalize_consolidated_output(consolidated_data: Dict, results: List[Dict], args, logger):
    """Finalize and save consolidated output files"""
    try:
        output_dir = Path(args.output_dir) / getattr(args, 'consolidated_subdir', 'consolidated')

        # Calculate final summary statistics
        total_papers = len(results)
        unique_metabolites = len(consolidated_data['all_metabolites'])
        unique_matches = len(consolidated_data['all_matches'])
        avg_detection_rate = sum(r['matching_result']['detection_rate'] for r in results) / total_papers if total_papers > 0 else 0

        consolidated_data['summary_stats'].update({
            'total_papers': total_papers,
            'total_unique_metabolites': unique_metabolites,
            'total_unique_matches': unique_matches,
            'average_detection_rate': avg_detection_rate,
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

        # Save consolidated JSON (append mode)
        consolidated_json = output_dir / "consolidated_results.json"
        save_consolidated_json(consolidated_data, consolidated_json, logger)

        # Save consolidated CSV (append mode)
        if args.export_format in ['csv', 'all']:
            consolidated_csv = output_dir / "consolidated_metabolites.csv"
            save_consolidated_csv(consolidated_data, consolidated_csv, logger)

        # Save consolidated Excel (append mode)
        if args.export_format in ['xlsx', 'all']:
            consolidated_xlsx = output_dir / "consolidated_results.xlsx"
            save_consolidated_xlsx(consolidated_data, consolidated_xlsx, logger)

        # Save summary report
        summary_file = output_dir / "processing_summary.json"
        save_processing_summary(consolidated_data, summary_file, logger)

    except Exception as e:
        logger.error(f"❌ Failed to finalize consolidated output: {e}")
        raise

def save_consolidated_json(consolidated_data: Dict, output_file: Path, logger):
    """Save consolidated JSON with append mode"""
    try:
        # Load existing data if file exists
        existing_data = {'processing_runs': []}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                if 'processing_runs' not in existing_data:
                    existing_data = {'processing_runs': [existing_data]}  # Convert old format
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing consolidated data: {e}")

        # Add new run data
        new_run = {
            'timestamp': consolidated_data['summary_stats']['processing_timestamp'],
            'summary_stats': consolidated_data['summary_stats'],
            'paper_results': consolidated_data['paper_results'],
            'all_unique_metabolites': sorted(list(consolidated_data['all_metabolites'])),
            'all_unique_matches': sorted(list(consolidated_data['all_matches']))
        }

        existing_data['processing_runs'].append(new_run)

        # Save updated data
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)

        logger.info(f"💾 Saved consolidated JSON: {output_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save consolidated JSON: {e}")

def save_consolidated_csv(consolidated_data: Dict, output_file: Path, logger):
    """Save consolidated CSV with append mode"""
    try:
        # Prepare data for CSV
        csv_data = []
        timestamp = consolidated_data['summary_stats']['processing_timestamp']

        for paper_result in consolidated_data['paper_results']:
            for metabolite in paper_result['metabolites']:
                csv_data.append({
                    'Processing_Timestamp': timestamp,
                    'Paper_Name': paper_result['paper_name'],
                    'Paper_File': paper_result['paper_file'],
                    'Metabolite': metabolite,
                    'In_Database': metabolite in paper_result['matched_biomarkers'],
                    'Paper_Detection_Rate': paper_result['detection_rate'],
                    'Paper_Precision': paper_result['precision'],
                    'Paper_Recall': paper_result['recall'],
                    'Paper_F1_Score': paper_result['f1_score']
                })

        df = pd.DataFrame(csv_data)

        # Append to existing file or create new
        if output_file.exists():
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)

        logger.info(f"💾 Saved consolidated CSV: {output_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save consolidated CSV: {e}")

def save_consolidated_xlsx(consolidated_data: Dict, output_file: Path, logger):
    """Save consolidated Excel with multiple sheets"""
    try:
        # Load existing workbook or create new
        existing_data = []
        if output_file.exists():
            try:
                existing_df = pd.read_excel(output_file, sheet_name='All_Runs')
                existing_data = existing_df.to_dict('records')
            except Exception:
                pass  # File doesn't exist or is corrupted

        # Prepare current run data
        current_run_data = []
        timestamp = consolidated_data['summary_stats']['processing_timestamp']

        for paper_result in consolidated_data['paper_results']:
            current_run_data.append({
                'Processing_Timestamp': timestamp,
                'Paper_Name': paper_result['paper_name'],
                'Metabolites_Count': paper_result['metabolites_count'],
                'Matches_Count': paper_result['matches_count'],
                'Detection_Rate': paper_result['detection_rate'],
                'Precision': paper_result['precision'],
                'Recall': paper_result['recall'],
                'F1_Score': paper_result['f1_score'],
                'Processing_Time': paper_result['processing_time']
            })

        # Combine with existing data
        all_data = existing_data + current_run_data

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # All runs sheet
            pd.DataFrame(all_data).to_excel(writer, sheet_name='All_Runs', index=False)

            # Current run details
            pd.DataFrame(current_run_data).to_excel(writer, sheet_name='Current_Run', index=False)

            # Summary statistics
            summary_df = pd.DataFrame([consolidated_data['summary_stats']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # All unique metabolites
            metabolites_df = pd.DataFrame({
                'Metabolite': sorted(list(consolidated_data['all_metabolites'])),
                'In_Database': [m in consolidated_data['all_matches'] for m in sorted(list(consolidated_data['all_metabolites']))]
            })
            metabolites_df.to_excel(writer, sheet_name='All_Metabolites', index=False)

        logger.info(f"💾 Saved consolidated Excel: {output_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save consolidated Excel: {e}")

def save_processing_summary(consolidated_data: Dict, output_file: Path, logger):
    """Save processing summary with append mode"""
    try:
        # Load existing summaries
        existing_summaries = []
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    existing_summaries = data.get('processing_summaries', [])
            except Exception:
                pass

        # Add current summary
        current_summary = {
            'timestamp': consolidated_data['summary_stats']['processing_timestamp'],
            'summary_stats': consolidated_data['summary_stats'],
            'paper_count': len(consolidated_data['paper_results']),
            'top_metabolites': list(consolidated_data['all_metabolites'])[:20],  # Top 20
            'paper_names': [pr['paper_name'] for pr in consolidated_data['paper_results']]
        }

        existing_summaries.append(current_summary)

        # Save updated summaries
        with open(output_file, 'w') as f:
            json.dump({'processing_summaries': existing_summaries}, f, indent=2)

        logger.info(f"💾 Saved processing summary: {output_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save processing summary: {e}")

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

def get_output_filename(input_file: str, args, extension: str = '.json') -> Path:
    """Generate output filename based on input file and arguments with timestamp support"""
    input_path = Path(input_file)
    output_dir = Path(args.output_dir)

    # Create base filename
    base_name = input_path.stem
    if args.output_prefix:
        base_name = f"{args.output_prefix}_{base_name}"

    # Add timestamp to preserve old files (unless explicitly disabled)
    use_timestamp = getattr(args, 'timestamp_files', True) and not getattr(args, 'no_timestamp', False)

    if use_timestamp:
        if hasattr(args, 'custom_timestamp') and args.custom_timestamp:
            # Use custom timestamp
            timestamp = args.custom_timestamp
        else:
            # Generate timestamp using specified format
            timestamp_format = getattr(args, 'timestamp_format', '%Y%m%d_%H%M%S')
            timestamp = time.strftime(timestamp_format)

        base_name = f"{base_name}_{timestamp}"

    return output_dir / f"{base_name}_results{extension}"

def save_results(result: Dict[str, Any], input_file: str, args, logger):
    """Save results to output files with timestamp support"""
    try:
        # Generate base output filename with timestamp
        output_file = get_output_filename(input_file, args, '.json')

        # Save JSON results
        if args.export_format in ['json', 'all']:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"💾 Saved JSON results: {output_file}")

        # Save CSV results
        if args.export_format in ['csv', 'all']:
            csv_file = get_output_filename(input_file, args, '.csv')
            save_csv_results(result, csv_file, logger)

        # Save Excel results
        if args.export_format in ['xlsx', 'all']:
            xlsx_file = get_output_filename(input_file, args, '.xlsx')
            save_xlsx_results(result, xlsx_file, logger)

        # Save timing analysis if requested
        if args.save_timing:
            timing_file = get_timestamped_filename(output_file, '_timing', '.json')
            save_timing_analysis(result, timing_file, logger)

        # Save raw responses if requested
        if args.save_raw_responses:
            raw_file = get_timestamped_filename(output_file, '_raw_responses', '.json')
            save_raw_responses(result, raw_file, logger)

        # Save chunks if requested
        if args.save_chunks and isinstance(result['chunks'], list):
            chunks_dir = get_timestamped_filename(output_file, '_chunks', '')
            save_chunks(result['chunks'], chunks_dir, logger)

    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise

def get_timestamped_filename(base_file: Path, suffix: str, extension: str) -> Path:
    """Generate timestamped filename with suffix"""
    if extension:
        return base_file.with_name(f"{base_file.stem}{suffix}{extension}")
    else:
        return base_file.with_name(f"{base_file.stem}{suffix}")

def save_raw_responses(result: Dict[str, Any], raw_file: Path, logger):
    """Save raw LLM responses for debugging"""
    try:
        raw_responses = []

        if 'extraction_result' in result and 'chunk_results' in result['extraction_result']:
            for chunk_result in result['extraction_result']['chunk_results']:
                if 'raw_response' in chunk_result and chunk_result['raw_response']:
                    raw_responses.append({
                        'chunk_id': chunk_result['chunk_id'],
                        'success': chunk_result['success'],
                        'processing_time': chunk_result['processing_time'],
                        'raw_response': chunk_result['raw_response']
                    })

        with open(raw_file, 'w') as f:
            json.dump(raw_responses, f, indent=2)

        logger.info(f"💾 Saved raw responses: {raw_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save raw responses: {e}")

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
    """Generate summary for batch processing with timestamp"""
    try:
        # Generate timestamped batch summary filename
        use_timestamp = getattr(args, 'timestamp_files', True) and not getattr(args, 'no_timestamp', False)

        if use_timestamp:
            if hasattr(args, 'custom_timestamp') and args.custom_timestamp:
                timestamp = args.custom_timestamp
            else:
                timestamp_format = getattr(args, 'timestamp_format', '%Y%m%d_%H%M%S')
                timestamp = time.strftime(timestamp_format)
            summary_file = Path(args.output_dir) / f"batch_summary_{timestamp}.json"
        else:
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
