#!/usr/bin/env python3
"""
Comprehensive Pipeline Timing Analysis
This script runs the complete FOODB pipeline with detailed timing for each component
and saves a comprehensive timing report to an output file.
"""

import sys
import os
import csv
import json
import time
import re
from pathlib import Path
from typing import List, Set, Dict, Tuple
from datetime import datetime

# Add the FOODB_LLM_pipeline directory to the path
sys.path.append('FOODB_LLM_pipeline')

class PipelineTimer:
    """Comprehensive timing tracker for the entire pipeline"""

    def __init__(self):
        self.timings = {}
        self.step_details = {}
        self.start_times = {}
        self.total_start_time = None
        self.chunk_timings = []
        self.batch_timings = []

    def start_total(self):
        """Start timing the entire pipeline"""
        self.total_start_time = time.time()

    def start_step(self, step_name: str, details: str = ""):
        """Start timing a specific step"""
        self.start_times[step_name] = time.time()
        self.step_details[step_name] = details

    def end_step(self, step_name: str) -> float:
        """End timing a step and return duration"""
        if step_name in self.start_times:
            duration = time.time() - self.start_times[step_name]
            self.timings[step_name] = duration
            return duration
        return 0.0

    def add_chunk_timing(self, chunk_num: int, total_time: float, api_time: float,
                        metabolites_found: int, chunk_size: int):
        """Add timing for individual chunk processing"""
        self.chunk_timings.append({
            'chunk_number': chunk_num,
            'total_time': total_time,
            'api_time': api_time,
            'processing_time': total_time - api_time,
            'metabolites_found': metabolites_found,
            'chunk_size': chunk_size,
            'efficiency': metabolites_found / total_time if total_time > 0 else 0
        })

    def add_batch_timing(self, batch_num: int, batch_size: int, batch_time: float):
        """Add timing for batch processing"""
        self.batch_timings.append({
            'batch_number': batch_num,
            'batch_size': batch_size,
            'batch_time': batch_time,
            'avg_chunk_time': batch_time / batch_size if batch_size > 0 else 0
        })

    def get_total_time(self) -> float:
        """Get total elapsed time"""
        if self.total_start_time:
            return time.time() - self.total_start_time
        return 0.0

    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive timing report"""
        total_time = self.get_total_time()

        # Calculate step statistics
        step_stats = []
        for step_name, duration in self.timings.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            step_stats.append({
                'step_name': step_name,
                'duration': duration,
                'percentage': percentage,
                'details': self.step_details.get(step_name, "")
            })

        # Calculate chunk statistics
        chunk_stats = {
            'total_chunks': len(self.chunk_timings),
            'total_api_time': sum(c['api_time'] for c in self.chunk_timings),
            'total_processing_time': sum(c['processing_time'] for c in self.chunk_timings),
            'total_metabolites': sum(c['metabolites_found'] for c in self.chunk_timings),
            'avg_chunk_time': sum(c['total_time'] for c in self.chunk_timings) / len(self.chunk_timings) if self.chunk_timings else 0,
            'avg_api_time': sum(c['api_time'] for c in self.chunk_timings) / len(self.chunk_timings) if self.chunk_timings else 0,
            'avg_metabolites_per_chunk': sum(c['metabolites_found'] for c in self.chunk_timings) / len(self.chunk_timings) if self.chunk_timings else 0,
            'fastest_chunk': min(self.chunk_timings, key=lambda x: x['total_time']) if self.chunk_timings else None,
            'slowest_chunk': max(self.chunk_timings, key=lambda x: x['total_time']) if self.chunk_timings else None,
            'most_efficient_chunk': max(self.chunk_timings, key=lambda x: x['efficiency']) if self.chunk_timings else None
        }

        # Calculate batch statistics
        batch_stats = {
            'total_batches': len(self.batch_timings),
            'avg_batch_time': sum(b['batch_time'] for b in self.batch_timings) / len(self.batch_timings) if self.batch_timings else 0,
            'fastest_batch': min(self.batch_timings, key=lambda x: x['batch_time']) if self.batch_timings else None,
            'slowest_batch': max(self.batch_timings, key=lambda x: x['batch_time']) if self.batch_timings else None
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'step_breakdown': step_stats,
            'chunk_statistics': chunk_stats,
            'batch_statistics': batch_stats,
            'detailed_chunk_timings': self.chunk_timings,
            'detailed_batch_timings': self.batch_timings
        }

def extract_pdf_with_timing(pdf_path: str, timer: PipelineTimer) -> str:
    """Extract PDF text with detailed timing"""
    timer.start_step("pdf_dependency_check", "Installing PyPDF2 if needed")

    try:
        import PyPDF2
        timer.end_step("pdf_dependency_check")
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
        import PyPDF2
        timer.end_step("pdf_dependency_check")

    timer.start_step("pdf_text_extraction", f"Extracting text from {pdf_path}")

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            page_count = len(pdf_reader.pages)
            timer.step_details["pdf_text_extraction"] += f" ({page_count} pages)"

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as e:
                    continue

            timer.end_step("pdf_text_extraction")
            return text

    except Exception as e:
        timer.end_step("pdf_text_extraction")
        return ""

def load_expected_data_with_timing(csv_path: str, timer: PipelineTimer) -> List[str]:
    """Load expected metabolites with timing"""
    timer.start_step("csv_loading", f"Loading expected metabolites from {csv_path}")

    try:
        metabolites = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header

            for row in csv_reader:
                if row and row[0].strip():
                    metabolites.append(row[0].strip())

        timer.step_details["csv_loading"] += f" ({len(metabolites)} metabolites)"
        timer.end_step("csv_loading")
        return metabolites

    except Exception as e:
        timer.end_step("csv_loading")
        return []

def initialize_wrapper_with_timing(timer: PipelineTimer):
    """Initialize LLM wrapper with timing"""
    timer.start_step("wrapper_initialization", "Loading LLM wrapper and model")

    try:
        from llm_wrapper import LLMWrapper
        wrapper = LLMWrapper()

        model_name = wrapper.current_model.get('model_name', 'Unknown')
        timer.step_details["wrapper_initialization"] += f" ({model_name})"

        timer.end_step("wrapper_initialization")
        return wrapper

    except Exception as e:
        timer.end_step("wrapper_initialization")
        return None

def chunk_text_with_timing(text: str, timer: PipelineTimer, chunk_size: int = 1500) -> List[str]:
    """Chunk text with timing"""
    timer.start_step("text_chunking", f"Splitting text into chunks (size: {chunk_size})")

    words = text.split()
    chunks = []

    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    timer.step_details["text_chunking"] += f" ({len(chunks)} chunks created)"
    timer.end_step("text_chunking")
    return chunks

def extract_metabolites_with_comprehensive_timing(wrapper, text_chunks: List[str],
                                                timer: PipelineTimer, batch_size: int = 5) -> List[str]:
    """Extract metabolites with comprehensive timing for each chunk and batch"""
    timer.start_step("metabolite_extraction", f"Processing {len(text_chunks)} chunks in batches")

    all_metabolites = []

    # Process in batches
    for batch_start in range(0, len(text_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(text_chunks))
        batch_chunks = text_chunks[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        print(f"📦 Processing batch {batch_num}/{(len(text_chunks)-1)//batch_size + 1} "
              f"(chunks {batch_start+1}-{batch_end})")

        batch_start_time = time.time()

        # Process each chunk in the batch
        for i, chunk in enumerate(batch_chunks):
            chunk_num = batch_start + i + 1
            print(f"  📄 Chunk {chunk_num}/{len(text_chunks)}...", end=" ")

            # Enhanced prompt for metabolite extraction
            prompt = f"""Extract ALL chemical compounds, metabolites, and biomarkers from this wine research text. Include:

1. Wine phenolic compounds (anthocyanins, flavonoids, phenolic acids)
2. Urinary metabolites and biomarkers
3. Chemical names with modifications (glucoside, glucuronide, sulfate)
4. Specific compound names mentioned
5. Metabolic derivatives and conjugates

Text: {chunk}

List each compound on a separate line (no numbering):"""

            try:
                chunk_start_time = time.time()
                api_start_time = time.time()
                response = wrapper.generate_single(prompt, max_tokens=600)
                api_end_time = time.time()

                # Parse metabolites from response
                lines = response.strip().split('\n')
                chunk_metabolites = []

                for line in lines:
                    line = line.strip()
                    # Clean up the line
                    line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                    line = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
                    line = re.sub(r'^[A-Z]\.\s*', '', line)  # Remove letter numbering

                    # Skip headers and non-compound lines
                    skip_patterns = [
                        'here are', 'the following', 'compounds found', 'metabolites',
                        'biomarkers', 'chemical', 'wine', 'urinary', 'list', 'include',
                        'based on', 'from the text', 'mentioned in'
                    ]

                    if (line and len(line) > 3 and
                        not any(pattern in line.lower() for pattern in skip_patterns) and
                        not line.startswith(('Here', 'The', 'List', 'Compounds', 'Based on', 'From the'))):
                        chunk_metabolites.append(line)

                all_metabolites.extend(chunk_metabolites)

                chunk_end_time = time.time()
                chunk_total_time = chunk_end_time - chunk_start_time
                chunk_api_time = api_end_time - api_start_time

                # Record chunk timing
                timer.add_chunk_timing(
                    chunk_num=chunk_num,
                    total_time=chunk_total_time,
                    api_time=chunk_api_time,
                    metabolites_found=len(chunk_metabolites),
                    chunk_size=len(chunk)
                )

                print(f"{len(chunk_metabolites)} metabolites ({chunk_total_time:.2f}s)")

            except Exception as e:
                chunk_end_time = time.time()
                chunk_total_time = chunk_end_time - chunk_start_time

                # Record failed chunk timing
                timer.add_chunk_timing(
                    chunk_num=chunk_num,
                    total_time=chunk_total_time,
                    api_time=0,
                    metabolites_found=0,
                    chunk_size=len(chunk)
                )

                print(f"❌ Error: {e}")
                continue

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        # Record batch timing
        timer.add_batch_timing(
            batch_num=batch_num,
            batch_size=len(batch_chunks),
            batch_time=batch_time
        )

        print(f"  ✅ Batch completed in {batch_time:.2f}s")

        # Small delay between batches to be respectful to API
        if batch_end < len(text_chunks):
            time.sleep(1)

    # Remove duplicates
    unique_metabolites = []
    seen = set()
    for metabolite in all_metabolites:
        metabolite_lower = metabolite.lower().strip()
        if metabolite_lower not in seen and len(metabolite_lower) > 2:
            seen.add(metabolite_lower)
            unique_metabolites.append(metabolite.strip())

    timer.step_details["metabolite_extraction"] += f" ({len(unique_metabolites)} unique metabolites)"
    timer.end_step("metabolite_extraction")

    return unique_metabolites

def analyze_results_with_timing(extracted: List[str], expected: List[str], timer: PipelineTimer) -> Dict:
    """Analyze results with timing"""
    timer.start_step("result_analysis", f"Analyzing {len(extracted)} extracted vs {len(expected)} expected")

    # Normalize names for comparison
    def normalize_name(name: str) -> str:
        return re.sub(r'[^\w\s-]', '', name.lower()).strip()

    extracted_normalized = {normalize_name(m): m for m in extracted}
    expected_normalized = {normalize_name(m): m for m in expected}

    # Find matches
    matches = []
    for norm_expected, original_expected in expected_normalized.items():
        for norm_extracted, original_extracted in extracted_normalized.items():
            if (norm_expected == norm_extracted or
                norm_expected in norm_extracted or
                norm_extracted in norm_expected):
                matches.append({
                    'expected': original_expected,
                    'extracted': original_extracted,
                    'match_type': 'exact' if norm_expected == norm_extracted else 'partial'
                })
                break

    # Calculate metrics
    total_expected = len(expected)
    total_extracted = len(extracted)
    total_matches = len(matches)

    precision = total_matches / total_extracted if total_extracted > 0 else 0
    recall = total_matches / total_expected if total_expected > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'total_expected': total_expected,
        'total_extracted': total_extracted,
        'total_matches': total_matches,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matches': matches
    }

    timer.step_details["result_analysis"] += f" ({total_matches} matches found)"
    timer.end_step("result_analysis")

    return results

def save_timing_report(timing_report: Dict, results: Dict):
    """Save comprehensive timing report to output files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON report
    json_filename = f"Pipeline_Timing_Report_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'timing_analysis': timing_report,
            'pipeline_results': results
        }, f, indent=2, default=str)

    # Save human-readable report
    txt_filename = f"Pipeline_Timing_Report_{timestamp}.txt"
    with open(txt_filename, 'w') as f:
        write_human_readable_report(f, timing_report, results)

    return json_filename, txt_filename

def write_human_readable_report(f, timing_report: Dict, results: Dict):
    """Write human-readable timing report"""
    f.write("=" * 80 + "\n")
    f.write("                    FOODB PIPELINE COMPREHENSIVE TIMING REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Generated: {timing_report['timestamp']}\n")
    f.write(f"Total Pipeline Time: {timing_report['total_time']:.2f} seconds\n\n")

    # Step breakdown
    f.write("PIPELINE STEP BREAKDOWN:\n")
    f.write("-" * 50 + "\n")
    for i, step in enumerate(timing_report['step_breakdown'], 1):
        f.write(f"{i:2d}. {step['step_name'].replace('_', ' ').title():<25} "
                f"{step['duration']:8.3f}s ({step['percentage']:5.1f}%)\n")
        f.write(f"    Details: {step['details']}\n\n")

    # Chunk statistics
    chunk_stats = timing_report['chunk_statistics']
    f.write("CHUNK PROCESSING STATISTICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Chunks Processed: {chunk_stats['total_chunks']}\n")
    f.write(f"Total API Time: {chunk_stats['total_api_time']:.2f}s\n")
    f.write(f"Total Processing Time: {chunk_stats['total_processing_time']:.2f}s\n")
    f.write(f"Average Chunk Time: {chunk_stats['avg_chunk_time']:.3f}s\n")
    f.write(f"Average API Time: {chunk_stats['avg_api_time']:.3f}s\n")
    f.write(f"Total Metabolites Found: {chunk_stats['total_metabolites']}\n")
    f.write(f"Average Metabolites per Chunk: {chunk_stats['avg_metabolites_per_chunk']:.1f}\n\n")

    if chunk_stats['fastest_chunk']:
        f.write(f"Fastest Chunk: #{chunk_stats['fastest_chunk']['chunk_number']} "
                f"({chunk_stats['fastest_chunk']['total_time']:.3f}s)\n")
    if chunk_stats['slowest_chunk']:
        f.write(f"Slowest Chunk: #{chunk_stats['slowest_chunk']['chunk_number']} "
                f"({chunk_stats['slowest_chunk']['total_time']:.3f}s)\n")
    if chunk_stats['most_efficient_chunk']:
        f.write(f"Most Efficient: #{chunk_stats['most_efficient_chunk']['chunk_number']} "
                f"({chunk_stats['most_efficient_chunk']['efficiency']:.1f} metabolites/s)\n\n")

    # Batch statistics
    batch_stats = timing_report['batch_statistics']
    f.write("BATCH PROCESSING STATISTICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Batches: {batch_stats['total_batches']}\n")
    f.write(f"Average Batch Time: {batch_stats['avg_batch_time']:.2f}s\n")
    if batch_stats['fastest_batch']:
        f.write(f"Fastest Batch: #{batch_stats['fastest_batch']['batch_number']} "
                f"({batch_stats['fastest_batch']['batch_time']:.2f}s)\n")
    if batch_stats['slowest_batch']:
        f.write(f"Slowest Batch: #{batch_stats['slowest_batch']['batch_number']} "
                f"({batch_stats['slowest_batch']['batch_time']:.2f}s)\n\n")

    # Performance results
    f.write("PIPELINE PERFORMANCE RESULTS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Expected Metabolites: {results['total_expected']}\n")
    f.write(f"Extracted Metabolites: {results['total_extracted']}\n")
    f.write(f"Matches Found: {results['total_matches']}\n")
    f.write(f"Precision: {results['precision']:.1%}\n")
    f.write(f"Recall: {results['recall']:.1%}\n")
    f.write(f"F1-Score: {results['f1_score']:.1%}\n\n")

    # Detailed chunk timings
    f.write("DETAILED CHUNK TIMINGS:\n")
    f.write("-" * 40 + "\n")
    f.write("Chunk  Total(s)  API(s)  Proc(s)  Metabolites  Efficiency(met/s)\n")
    f.write("-" * 70 + "\n")

    for chunk in timing_report['detailed_chunk_timings']:
        f.write(f"{chunk['chunk_number']:5d}  "
                f"{chunk['total_time']:7.3f}  "
                f"{chunk['api_time']:6.3f}  "
                f"{chunk['processing_time']:7.3f}  "
                f"{chunk['metabolites_found']:10d}  "
                f"{chunk['efficiency']:13.1f}\n")

    # Performance analysis
    f.write("\n" + "=" * 80 + "\n")
    f.write("PERFORMANCE ANALYSIS:\n")
    f.write("=" * 80 + "\n\n")

    total_time = timing_report['total_time']
    api_time = chunk_stats['total_api_time']
    processing_time = chunk_stats['total_processing_time']

    f.write(f"Time Distribution:\n")
    f.write(f"  API Calls: {api_time:.2f}s ({api_time/total_time*100:.1f}%)\n")
    f.write(f"  Local Processing: {processing_time:.2f}s ({processing_time/total_time*100:.1f}%)\n")
    f.write(f"  Setup/Analysis: {total_time - api_time - processing_time:.2f}s "
            f"({(total_time - api_time - processing_time)/total_time*100:.1f}%)\n\n")

    f.write(f"Throughput Metrics:\n")
    f.write(f"  Metabolites per second: {chunk_stats['total_metabolites']/total_time:.1f}\n")
    f.write(f"  Chunks per second: {chunk_stats['total_chunks']/total_time:.2f}\n")
    f.write(f"  API calls per second: {chunk_stats['total_chunks']/api_time:.2f}\n\n")

    f.write(f"Efficiency Analysis:\n")
    f.write(f"  API efficiency: {api_time/total_time*100:.1f}% of total time\n")
    f.write(f"  Processing overhead: {processing_time/api_time*100:.1f}% of API time\n")
    f.write(f"  Average metabolites per API call: {chunk_stats['avg_metabolites_per_chunk']:.1f}\n")

def run_comprehensive_timing_analysis():
    """Run the complete pipeline with comprehensive timing analysis"""
    print("🕐 FOODB Pipeline Comprehensive Timing Analysis")
    print("=" * 60)

    # Initialize timer
    timer = PipelineTimer()
    timer.start_total()

    # File paths
    pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
    csv_path = "urinary_wine_biomarkers.csv"

    try:
        # Step 1: Extract PDF text
        print("📄 Step 1: PDF Text Extraction")
        pdf_text = extract_pdf_with_timing(pdf_path, timer)
        if not pdf_text:
            print("❌ Failed to extract PDF text")
            return
        print(f"✅ Extracted {len(pdf_text)} characters")

        # Step 2: Load expected metabolites
        print("\n📋 Step 2: Loading Expected Data")
        expected_metabolites = load_expected_data_with_timing(csv_path, timer)
        if not expected_metabolites:
            print("❌ Failed to load expected metabolites")
            return
        print(f"✅ Loaded {len(expected_metabolites)} expected metabolites")

        # Step 3: Initialize wrapper
        print("\n🔬 Step 3: Wrapper Initialization")
        wrapper = initialize_wrapper_with_timing(timer)
        if not wrapper:
            print("❌ Failed to initialize wrapper")
            return
        print(f"✅ Initialized {wrapper.current_model.get('model_name', 'Unknown')} model")

        # Step 4: Chunk text
        print("\n📝 Step 4: Text Chunking")
        text_chunks = chunk_text_with_timing(pdf_text, timer)
        print(f"✅ Created {len(text_chunks)} chunks")

        # Step 5: Extract metabolites with comprehensive timing
        print(f"\n🧬 Step 5: Metabolite Extraction ({len(text_chunks)} chunks)")
        extracted_metabolites = extract_metabolites_with_comprehensive_timing(
            wrapper, text_chunks, timer, batch_size=5
        )
        print(f"✅ Extracted {len(extracted_metabolites)} unique metabolites")

        # Step 6: Analyze results
        print("\n🔍 Step 6: Result Analysis")
        results = analyze_results_with_timing(extracted_metabolites, expected_metabolites, timer)
        print(f"✅ Found {results['total_matches']} matches")

        # Generate comprehensive timing report
        print("\n📊 Generating Comprehensive Timing Report...")
        timing_report = timer.generate_comprehensive_report()

        # Save reports
        json_file, txt_file = save_timing_report(timing_report, results)

        # Display summary
        print(f"\n📊 PIPELINE TIMING SUMMARY")
        print("=" * 40)
        print(f"Total Pipeline Time: {timing_report['total_time']:.2f}s")
        print(f"Chunks Processed: {timing_report['chunk_statistics']['total_chunks']}")
        print(f"Total API Time: {timing_report['chunk_statistics']['total_api_time']:.2f}s")
        print(f"Average per Chunk: {timing_report['chunk_statistics']['avg_chunk_time']:.3f}s")
        print(f"Metabolites Found: {timing_report['chunk_statistics']['total_metabolites']}")
        print(f"Detection Rate: {results['recall']:.1%}")
        print(f"Precision: {results['precision']:.1%}")
        print(f"F1-Score: {results['f1_score']:.1%}")

        print(f"\n💾 Reports Saved:")
        print(f"  JSON Report: {json_file}")
        print(f"  Text Report: {txt_file}")

        print(f"\n🎉 Comprehensive timing analysis complete!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_timing_analysis()