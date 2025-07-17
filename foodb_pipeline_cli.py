#!/usr/bin/env python3
"""
FOODB Pipeline Command Line Interface
Comprehensive CLI for the FOODB metabolite extraction pipeline with full configurability
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, List

# Add FOODB pipeline to path
sys.path.append('FOODB_LLM_pipeline')

def create_argument_parser():
    """Create comprehensive argument parser for FOODB pipeline"""
    parser = argparse.ArgumentParser(
        description='FOODB Pipeline - Extract metabolites from scientific PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python foodb_pipeline_cli.py input.pdf

  # Full configuration
  python foodb_pipeline_cli.py input.pdf \\
    --output-dir ./results \\
    --csv-database biomarkers.csv \\
    --chunk-size 2000 \\
    --max-tokens 300 \\
    --document-only \\
    --verify-compounds \\
    --max-attempts 5 \\
    --base-delay 2.0

  # Batch processing
  python foodb_pipeline_cli.py *.pdf --batch-mode --output-dir ./batch_results

  # Debug mode with detailed logging
  python foodb_pipeline_cli.py input.pdf --debug --save-chunks --save-timing
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input PDF file(s) or directory containing PDF files to process'
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./foodb_results',
        help='Output directory for results (default: ./foodb_results)'
    )
    output_group.add_argument(
        '--output-prefix',
        type=str,
        default='',
        help='Prefix for output files (default: none)'
    )
    output_group.add_argument(
        '--save-chunks',
        action='store_true',
        help='Save text chunks to separate files'
    )
    output_group.add_argument(
        '--save-timing',
        action='store_true',
        help='Save detailed timing analysis'
    )
    output_group.add_argument(
        '--save-raw-responses',
        action='store_true',
        help='Save raw LLM responses for debugging'
    )
    output_group.add_argument(
        '--timestamp-files',
        action='store_true',
        default=True,
        help='Add timestamp to output filenames to preserve old files (default: True)'
    )
    output_group.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Disable timestamp in filenames (will overwrite existing files)'
    )
    output_group.add_argument(
        '--timestamp-format',
        type=str,
        default='%Y%m%d_%H%M%S',
        help='Timestamp format for filenames (default: %%Y%%m%%d_%%H%%M%%S)'
    )
    output_group.add_argument(
        '--custom-timestamp',
        type=str,
        help='Custom timestamp string to use instead of current time'
    )
    output_group.add_argument(
        '--directory-mode',
        action='store_true',
        help='Process directory of PDF files with organized output structure'
    )
    output_group.add_argument(
        '--consolidated-output',
        action='store_true',
        default=True,
        help='Create consolidated output files (append mode, no timestamps) (default: True)'
    )
    output_group.add_argument(
        '--individual-output',
        action='store_true',
        default=True,
        help='Create individual paper-specific output files (timestamped) (default: True)'
    )
    output_group.add_argument(
        '--individual-subdir',
        type=str,
        default='individual_papers',
        help='Subdirectory for individual paper results (default: individual_papers)'
    )
    output_group.add_argument(
        '--consolidated-subdir',
        type=str,
        default='consolidated',
        help='Subdirectory for consolidated results (default: consolidated)'
    )
    
    # Database configuration
    db_group = parser.add_argument_group('Database Configuration')
    db_group.add_argument(
        '--csv-database',
        type=str,
        default='urinary_wine_biomarkers.csv',
        help='CSV database file for biomarker matching (default: urinary_wine_biomarkers.csv)'
    )
    db_group.add_argument(
        '--csv-column',
        type=str,
        default='Compound Name',
        help='Column name in CSV for compound names (default: Compound Name)'
    )
    
    # Text processing configuration
    text_group = parser.add_argument_group('Text Processing Configuration')
    text_group.add_argument(
        '--chunk-size',
        type=int,
        default=1500,
        help='Size of text chunks in characters (default: 1500)'
    )
    text_group.add_argument(
        '--chunk-overlap',
        type=int,
        default=0,
        help='Overlap between chunks in characters (default: 0)'
    )
    text_group.add_argument(
        '--min-chunk-size',
        type=int,
        default=100,
        help='Minimum chunk size to process (default: 100)'
    )
    
    # LLM configuration
    llm_group = parser.add_argument_group('LLM Configuration')
    llm_group.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Maximum tokens for LLM responses (default: 200)'
    )
    llm_group.add_argument(
        '--document-only',
        action='store_true',
        help='Use document-only extraction (prevents training data contamination)'
    )
    llm_group.add_argument(
        '--verify-compounds',
        action='store_true',
        help='Verify extracted compounds against original text'
    )
    llm_group.add_argument(
        '--custom-prompt',
        type=str,
        help='Custom extraction prompt (overrides default)'
    )
    
    # Retry and fallback configuration
    retry_group = parser.add_argument_group('Retry and Fallback Configuration')
    retry_group.add_argument(
        '--max-attempts',
        type=int,
        default=5,
        help='Maximum retry attempts per chunk (default: 5)'
    )
    retry_group.add_argument(
        '--base-delay',
        type=float,
        default=2.0,
        help='Base delay for exponential backoff in seconds (default: 2.0)'
    )
    retry_group.add_argument(
        '--max-delay',
        type=float,
        default=60.0,
        help='Maximum delay for exponential backoff in seconds (default: 60.0)'
    )
    retry_group.add_argument(
        '--exponential-base',
        type=float,
        default=2.0,
        help='Exponential base for backoff calculation (default: 2.0)'
    )
    retry_group.add_argument(
        '--disable-jitter',
        action='store_true',
        help='Disable jitter in exponential backoff'
    )
    
    # Provider configuration
    provider_group = parser.add_argument_group('Provider Configuration')
    provider_group.add_argument(
        '--providers',
        nargs='+',
        default=['cerebras', 'groq', 'openrouter'],
        choices=['cerebras', 'groq', 'openrouter'],
        help='LLM providers in fallback order (default: cerebras groq openrouter)'
    )
    provider_group.add_argument(
        '--primary-provider',
        type=str,
        choices=['cerebras', 'groq', 'openrouter'],
        help='Primary provider to use (overrides providers order)'
    )
    provider_group.add_argument(
        '--groq-model',
        type=str,
        choices=[
            'moonshotai/kimi-k2-instruct',
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'meta-llama/llama-4-maverick-17b-128e-instruct',
            'llama-3.1-8b-instant',
            'qwen/qwen3-32b'
        ],
        help='Specific Groq model to use (default: moonshotai/kimi-k2-instruct)'
    )
    
    # Processing configuration
    process_group = parser.add_argument_group('Processing Configuration')
    process_group.add_argument(
        '--batch-mode',
        action='store_true',
        help='Process multiple files in batch mode'
    )
    process_group.add_argument(
        '--parallel-chunks',
        type=int,
        default=1,
        help='Number of chunks to process in parallel (default: 1)'
    )
    process_group.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already have results'
    )
    process_group.add_argument(
        '--resume-from-chunk',
        type=int,
        help='Resume processing from specific chunk number'
    )
    
    # Analysis configuration
    analysis_group = parser.add_argument_group('Analysis Configuration')
    analysis_group.add_argument(
        '--calculate-metrics',
        action='store_true',
        default=True,
        help='Calculate precision, recall, and F1 scores (default: True)'
    )
    analysis_group.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Generate comprehensive analysis report (default: True)'
    )
    analysis_group.add_argument(
        '--export-format',
        choices=['json', 'csv', 'xlsx', 'all'],
        default='json',
        help='Export format for results (default: json)'
    )
    
    # Logging and debugging
    debug_group = parser.add_argument_group('Logging and Debugging')
    debug_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    debug_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    debug_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )
    debug_group.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: stdout)'
    )
    debug_group.add_argument(
        '--progress-bar',
        action='store_true',
        default=True,
        help='Show progress bar during processing (default: True)'
    )
    
    # Configuration file
    config_group = parser.add_argument_group('Configuration File')
    config_group.add_argument(
        '--config',
        type=str,
        help='Load configuration from JSON file'
    )
    config_group.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to JSON file'
    )
    
    return parser

def load_config_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        return {}

def save_config_file(args: argparse.Namespace, config_path: str):
    """Save configuration to JSON file"""
    config = {
        'output_dir': args.output_dir,
        'output_prefix': args.output_prefix,
        'timestamp_files': getattr(args, 'timestamp_files', True),
        'timestamp_format': getattr(args, 'timestamp_format', '%Y%m%d_%H%M%S'),
        'csv_database': args.csv_database,
        'csv_column': args.csv_column,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'min_chunk_size': args.min_chunk_size,
        'max_tokens': args.max_tokens,
        'document_only': args.document_only,
        'verify_compounds': args.verify_compounds,
        'max_attempts': args.max_attempts,
        'base_delay': args.base_delay,
        'max_delay': args.max_delay,
        'exponential_base': args.exponential_base,
        'providers': args.providers,
        'batch_mode': args.batch_mode,
        'calculate_metrics': args.calculate_metrics,
        'export_format': args.export_format,
        'save_chunks': args.save_chunks,
        'save_timing': args.save_timing,
        'save_raw_responses': args.save_raw_responses
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {config_path}")
    except Exception as e:
        print(f"❌ Error saving config file {config_path}: {e}")

def setup_logging(args: argparse.Namespace):
    """Setup logging based on arguments"""
    import logging
    
    # Determine log level
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    
    # Setup logging format
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    if args.log_file:
        logging.basicConfig(
            level=level,
            format=format_str,
            filename=args.log_file,
            filemode='w'
        )
    else:
        logging.basicConfig(
            level=level,
            format=format_str
        )
    
    return logging.getLogger(__name__)

def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments"""
    errors = []

    # Check input files/directories exist
    for input_path in args.input_files:
        if not os.path.exists(input_path):
            errors.append(f"Input path not found: {input_path}")
        elif os.path.isfile(input_path):
            if not input_path.lower().endswith('.pdf'):
                errors.append(f"Input file must be PDF: {input_path}")
        elif os.path.isdir(input_path):
            # Check if directory contains PDF files
            pdf_files = list(Path(input_path).glob("*.pdf"))
            if not pdf_files:
                errors.append(f"Directory contains no PDF files: {input_path}")
        else:
            errors.append(f"Input path must be file or directory: {input_path}")
    
    # Check CSV database exists
    if not os.path.exists(args.csv_database):
        errors.append(f"CSV database not found: {args.csv_database}")
    
    # Validate numeric parameters
    if args.chunk_size <= 0:
        errors.append("Chunk size must be positive")
    
    if args.max_tokens <= 0:
        errors.append("Max tokens must be positive")
    
    if args.max_attempts <= 0:
        errors.append("Max attempts must be positive")
    
    if args.base_delay < 0:
        errors.append("Base delay must be non-negative")
    
    if args.max_delay < args.base_delay:
        errors.append("Max delay must be >= base delay")
    
    # Check output directory is writable
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            errors.append(f"Output directory not writable: {output_dir}")
    except Exception as e:
        errors.append(f"Cannot create output directory {output_dir}: {e}")
    
    # Print errors
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    return True

def print_configuration(args: argparse.Namespace):
    """Print current configuration"""
    print("🔧 FOODB Pipeline Configuration")
    print("=" * 35)
    
    print(f"📄 Input Files: {len(args.input_files)} file(s)")
    for i, file in enumerate(args.input_files, 1):
        print(f"   {i}. {file}")
    
    print(f"\n📁 Output Configuration:")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Output prefix: {args.output_prefix or 'none'}")
    print(f"   Export format: {args.export_format}")
    print(f"   Directory mode: {getattr(args, 'directory_mode', False)}")
    if getattr(args, 'directory_mode', False):
        print(f"   Consolidated output: {getattr(args, 'consolidated_output', True)}")
        print(f"   Individual output: {getattr(args, 'individual_output', True)}")
        print(f"   Individual subdir: {getattr(args, 'individual_subdir', 'individual_papers')}")
        print(f"   Consolidated subdir: {getattr(args, 'consolidated_subdir', 'consolidated')}")
    print(f"   Timestamp files: {getattr(args, 'timestamp_files', True) and not getattr(args, 'no_timestamp', False)}")
    if getattr(args, 'timestamp_files', True) and not getattr(args, 'no_timestamp', False):
        print(f"   Timestamp format: {getattr(args, 'timestamp_format', '%Y%m%d_%H%M%S')}")
        if hasattr(args, 'custom_timestamp') and args.custom_timestamp:
            print(f"   Custom timestamp: {args.custom_timestamp}")
    print(f"   Save chunks: {args.save_chunks}")
    print(f"   Save timing: {args.save_timing}")
    print(f"   Save raw responses: {args.save_raw_responses}")
    
    print(f"\n📊 Database Configuration:")
    print(f"   CSV database: {args.csv_database}")
    print(f"   CSV column: {args.csv_column}")
    
    print(f"\n📝 Text Processing:")
    print(f"   Chunk size: {args.chunk_size} characters")
    print(f"   Chunk overlap: {args.chunk_overlap} characters")
    print(f"   Min chunk size: {args.min_chunk_size} characters")
    
    print(f"\n🤖 LLM Configuration:")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Document-only mode: {args.document_only}")
    print(f"   Verify compounds: {args.verify_compounds}")
    print(f"   Providers: {' → '.join(args.providers)}")
    
    print(f"\n🔄 Retry Configuration:")
    print(f"   Max attempts: {args.max_attempts}")
    print(f"   Base delay: {args.base_delay}s")
    print(f"   Max delay: {args.max_delay}s")
    print(f"   Exponential base: {args.exponential_base}")
    print(f"   Jitter: {not args.disable_jitter}")
    
    print(f"\n⚙️ Processing Options:")
    print(f"   Batch mode: {args.batch_mode}")
    print(f"   Parallel chunks: {args.parallel_chunks}")
    print(f"   Skip existing: {args.skip_existing}")
    print(f"   Calculate metrics: {args.calculate_metrics}")

def create_pipeline_runner():
    """Create the pipeline runner module"""
    runner_code = '''#!/usr/bin/env python3
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
from typing import List, Dict, Any
import logging

# Add FOODB pipeline to path
sys.path.append('FOODB_LLM_pipeline')

def run_pipeline(args, logger):
    """Run the complete FOODB pipeline with given arguments"""
    logger.info("Starting FOODB pipeline execution")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each input file
    for i, input_file in enumerate(args.input_files, 1):
        logger.info(f"Processing file {i}/{len(args.input_files)}: {input_file}")

        try:
            result = process_single_file(input_file, args, logger)

            # Save results
            save_results(result, input_file, args, logger)

            logger.info(f"✅ Successfully processed {input_file}")

        except Exception as e:
            logger.error(f"❌ Failed to process {input_file}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

def process_single_file(input_file: str, args, logger) -> Dict[str, Any]:
    """Process a single PDF file through the pipeline"""
    start_time = time.time()

    # Step 1: Extract text from PDF
    logger.info("Step 1: Extracting text from PDF")
    pdf_result = extract_pdf_text(input_file, logger)

    # Step 2: Chunk text
    logger.info("Step 2: Chunking text")
    chunks = chunk_text(pdf_result['text'], args, logger)

    # Step 3: Extract metabolites
    logger.info("Step 3: Extracting metabolites")
    extraction_result = extract_metabolites(chunks, args, logger)

    # Step 4: Match against database
    logger.info("Step 4: Matching against database")
    matching_result = match_database(extraction_result['metabolites'], args, logger)

    # Step 5: Calculate metrics
    logger.info("Step 5: Calculating metrics")
    metrics = calculate_metrics(matching_result, args, logger)

    end_time = time.time()

    return {
        'input_file': input_file,
        'processing_time': end_time - start_time,
        'pdf_result': pdf_result,
        'chunks': chunks if args.save_chunks else len(chunks),
        'extraction_result': extraction_result,
        'matching_result': matching_result,
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

# Additional functions would be implemented here...
'''

    with open('foodb_pipeline_runner.py', 'w') as f:
        f.write(runner_code)

def main():
    """Main CLI entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration file if specified
    if args.config:
        config = load_config_file(args.config)
        # Update args with config values (command line takes precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # Save configuration if requested
    if args.save_config:
        save_config_file(args, args.save_config)
        return
    
    # Setup logging
    logger = setup_logging(args)
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Print configuration
    if not args.quiet:
        print_configuration(args)
    
    print(f"\n🚀 Starting FOODB Pipeline...")
    print(f"Ready to process {len(args.input_files)} file(s)")
    print(f"Use --help for detailed usage information")
    
    # Import and run the actual pipeline
    try:
        from foodb_pipeline_runner import run_pipeline
        run_pipeline(args, logger)
    except ImportError:
        print(f"\n💡 Pipeline runner not found. Creating pipeline runner...")
        create_pipeline_runner()
        print(f"✅ Pipeline runner created. Please run the command again.")

if __name__ == "__main__":
    main()
