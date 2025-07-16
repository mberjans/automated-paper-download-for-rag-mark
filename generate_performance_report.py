#!/usr/bin/env python3
"""
FOODB Pipeline Performance Report Generator
This script generates a detailed performance report with timing for each pipeline step
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

class PerformanceTimer:
    """Class to track performance timing for each step"""
    
    def __init__(self):
        self.timings = {}
        self.step_details = {}
        self.start_times = {}
        self.total_start_time = None
    
    def start_total(self):
        """Start timing the entire process"""
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
    
    def get_total_time(self) -> float:
        """Get total elapsed time"""
        if self.total_start_time:
            return time.time() - self.total_start_time
        return 0.0
    
    def get_report(self) -> Dict:
        """Generate performance report"""
        total_time = self.get_total_time()
        
        report = {
            'total_time': total_time,
            'steps': [],
            'summary': {
                'total_steps': len(self.timings),
                'fastest_step': None,
                'slowest_step': None,
                'average_step_time': 0.0
            }
        }
        
        # Add step details
        for step_name, duration in self.timings.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            
            step_info = {
                'name': step_name,
                'duration': duration,
                'percentage': percentage,
                'details': self.step_details.get(step_name, "")
            }
            report['steps'].append(step_info)
        
        # Calculate summary statistics
        if self.timings:
            durations = list(self.timings.values())
            report['summary']['average_step_time'] = sum(durations) / len(durations)
            
            fastest = min(self.timings.items(), key=lambda x: x[1])
            slowest = max(self.timings.items(), key=lambda x: x[1])
            
            report['summary']['fastest_step'] = {'name': fastest[0], 'time': fastest[1]}
            report['summary']['slowest_step'] = {'name': slowest[0], 'time': slowest[1]}
        
        return report

def extract_pdf_with_timing(pdf_path: str, timer: PerformanceTimer) -> str:
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

def load_expected_data_with_timing(csv_path: str, timer: PerformanceTimer) -> List[str]:
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

def initialize_wrapper_with_timing(timer: PerformanceTimer):
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

def chunk_text_with_timing(text: str, timer: PerformanceTimer, chunk_size: int = 1500) -> List[str]:
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

def extract_metabolites_with_detailed_timing(wrapper, text_chunks: List[str], timer: PerformanceTimer) -> Tuple[List[str], Dict]:
    """Extract metabolites with detailed per-chunk timing"""
    timer.start_step("metabolite_extraction", f"Processing {len(text_chunks)} chunks")
    
    all_metabolites = []
    chunk_timings = []
    
    for i, chunk in enumerate(text_chunks, 1):
        chunk_start = time.time()
        
        # Create prompt for metabolite extraction
        prompt = f"""Extract all chemical compounds, metabolites, and biomarkers mentioned in this scientific text. Focus on:
- Wine-related compounds (anthocyanins, phenolic acids, etc.)
- Urinary metabolites and biomarkers
- Chemical names with specific structures
- Compounds with glucoside, glucuronide, sulfate modifications

Text: {chunk}

List all compounds found (one per line):"""
        
        try:
            api_start = time.time()
            response = wrapper.generate_single(prompt, max_tokens=400)
            api_end = time.time()
            
            # Parse metabolites from response
            lines = response.strip().split('\n')
            chunk_metabolites = []
            
            for line in lines:
                line = line.strip()
                # Clean up the line
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                line = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
                
                if line and len(line) > 3 and not line.startswith(('Here', 'The', 'List', 'Compounds')):
                    chunk_metabolites.append(line)
            
            all_metabolites.extend(chunk_metabolites)
            
            chunk_end = time.time()
            
            chunk_timing = {
                'chunk_number': i,
                'total_time': chunk_end - chunk_start,
                'api_time': api_end - api_start,
                'processing_time': (chunk_end - chunk_start) - (api_end - api_start),
                'metabolites_found': len(chunk_metabolites),
                'chunk_size': len(chunk)
            }
            chunk_timings.append(chunk_timing)
            
        except Exception as e:
            chunk_end = time.time()
            chunk_timing = {
                'chunk_number': i,
                'total_time': chunk_end - chunk_start,
                'api_time': 0,
                'processing_time': chunk_end - chunk_start,
                'metabolites_found': 0,
                'chunk_size': len(chunk),
                'error': str(e)
            }
            chunk_timings.append(chunk_timing)
    
    # Remove duplicates
    unique_metabolites = []
    seen = set()
    for metabolite in all_metabolites:
        metabolite_lower = metabolite.lower()
        if metabolite_lower not in seen:
            seen.add(metabolite_lower)
            unique_metabolites.append(metabolite)
    
    timer.step_details["metabolite_extraction"] += f" ({len(unique_metabolites)} unique metabolites)"
    timer.end_step("metabolite_extraction")
    
    return unique_metabolites, chunk_timings

def compare_results_with_timing(extracted: List[str], expected: List[str], timer: PerformanceTimer) -> Dict:
    """Compare results with timing"""
    timer.start_step("result_comparison", f"Comparing {len(extracted)} extracted vs {len(expected)} expected")
    
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
    
    timer.step_details["result_comparison"] += f" ({total_matches} matches found)"
    timer.end_step("result_comparison")
    
    return results

def generate_performance_report():
    """Generate comprehensive performance report"""
    print("📊 FOODB Pipeline Performance Report Generator")
    print("=" * 60)
    
    # Initialize timer
    timer = PerformanceTimer()
    timer.start_total()
    
    # File paths
    pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
    csv_path = "urinary_wine_biomarkers.csv"
    
    try:
        # Step 1: Extract PDF text
        pdf_text = extract_pdf_with_timing(pdf_path, timer)
        if not pdf_text:
            print("❌ Failed to extract PDF text")
            return
        
        # Step 2: Load expected metabolites
        expected_metabolites = load_expected_data_with_timing(csv_path, timer)
        if not expected_metabolites:
            print("❌ Failed to load expected metabolites")
            return
        
        # Step 3: Initialize wrapper
        wrapper = initialize_wrapper_with_timing(timer)
        if not wrapper:
            print("❌ Failed to initialize wrapper")
            return
        
        # Step 4: Chunk text
        text_chunks = chunk_text_with_timing(pdf_text, timer)
        
        # Step 5: Process subset for testing
        test_chunks = text_chunks[:5]  # Test first 5 chunks
        
        # Step 6: Extract metabolites with detailed timing
        extracted_metabolites, chunk_timings = extract_metabolites_with_detailed_timing(
            wrapper, test_chunks, timer
        )
        
        # Step 7: Compare results
        comparison_results = compare_results_with_timing(
            extracted_metabolites, expected_metabolites, timer
        )
        
        # Generate final report
        performance_report = timer.get_report()
        
        # Add chunk-level details
        performance_report['chunk_details'] = chunk_timings
        performance_report['results'] = comparison_results
        
        # Display report
        display_performance_report(performance_report)
        
        # Save detailed report
        save_performance_report(performance_report)
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()

def display_performance_report(report: Dict):
    """Display the performance report in a readable format"""
    print(f"\n📊 FOODB Pipeline Performance Report")
    print("=" * 60)
    print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total Pipeline Time: {report['total_time']:.2f} seconds")
    print()
    
    # Step-by-step breakdown
    print("📋 Step-by-Step Performance Breakdown:")
    print("-" * 50)
    
    for i, step in enumerate(report['steps'], 1):
        print(f"{i}. {step['name'].replace('_', ' ').title()}")
        print(f"   ⏱️ Time: {step['duration']:.3f}s ({step['percentage']:.1f}% of total)")
        print(f"   📝 Details: {step['details']}")
        print()
    
    # Performance summary
    summary = report['summary']
    print("📈 Performance Summary:")
    print("-" * 30)
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Average Step Time: {summary['average_step_time']:.3f}s")
    
    if summary['fastest_step']:
        print(f"Fastest Step: {summary['fastest_step']['name']} ({summary['fastest_step']['time']:.3f}s)")
    
    if summary['slowest_step']:
        print(f"Slowest Step: {summary['slowest_step']['name']} ({summary['slowest_step']['time']:.3f}s)")
    
    # Chunk-level performance
    if 'chunk_details' in report:
        print(f"\n🔍 Chunk-Level Performance:")
        print("-" * 35)
        
        chunk_details = report['chunk_details']
        total_api_time = sum(c['api_time'] for c in chunk_details)
        total_processing_time = sum(c['processing_time'] for c in chunk_details)
        total_metabolites = sum(c['metabolites_found'] for c in chunk_details)
        
        print(f"Chunks Processed: {len(chunk_details)}")
        print(f"Total API Time: {total_api_time:.3f}s")
        print(f"Total Processing Time: {total_processing_time:.3f}s")
        print(f"Average API Time per Chunk: {total_api_time/len(chunk_details):.3f}s")
        print(f"Total Metabolites Found: {total_metabolites}")
        print(f"Average Metabolites per Chunk: {total_metabolites/len(chunk_details):.1f}")
        
        print(f"\n📊 Per-Chunk Breakdown:")
        for chunk in chunk_details:
            print(f"  Chunk {chunk['chunk_number']}: {chunk['total_time']:.3f}s "
                  f"(API: {chunk['api_time']:.3f}s, Processing: {chunk['processing_time']:.3f}s) "
                  f"→ {chunk['metabolites_found']} metabolites")
    
    # Results summary
    if 'results' in report:
        results = report['results']
        print(f"\n🎯 Results Summary:")
        print("-" * 25)
        print(f"Expected Metabolites: {results['total_expected']}")
        print(f"Extracted Metabolites: {results['total_extracted']}")
        print(f"Matches Found: {results['total_matches']}")
        print(f"Precision: {results['precision']:.2%}")
        print(f"Recall: {results['recall']:.2%}")
        print(f"F1-Score: {results['f1_score']:.2%}")

def save_performance_report(report: Dict):
    """Save the performance report to a JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"foodb_performance_report_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed performance report saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save report: {e}")

if __name__ == "__main__":
    generate_performance_report()
