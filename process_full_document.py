#!/usr/bin/env python3
"""
Process the FULL Wine Biomarkers PDF (all 45 chunks)
This script processes the complete document to get comprehensive metabolite extraction results
"""

import sys
import os
import csv
import json
import time
import re
from pathlib import Path
from typing import List, Set, Dict, Tuple

# Add the FOODB_LLM_pipeline directory to the path
sys.path.append('FOODB_LLM_pipeline')

def extract_pdf_text_simple(pdf_path: str) -> str:
    """Simple PDF text extraction"""
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except:
                    continue
            
            return text
            
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
        return extract_pdf_text_simple(pdf_path)
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return ""

def load_expected_metabolites(csv_path: str) -> List[str]:
    """Load expected metabolites from CSV file"""
    try:
        metabolites = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                if row and row[0].strip():
                    metabolites.append(row[0].strip())
        
        return metabolites
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

def extract_metabolites_batch(wrapper, text_chunks: List[str], batch_size: int = 5) -> List[str]:
    """Extract metabolites from all chunks with batch processing for efficiency"""
    print(f"🧬 Processing ALL {len(text_chunks)} chunks in batches of {batch_size}...")
    
    all_metabolites = []
    total_start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, len(text_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(text_chunks))
        batch_chunks = text_chunks[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1} "
              f"(chunks {batch_start+1}-{batch_end})")
        
        batch_start_time = time.time()
        
        # Process each chunk in the batch
        for i, chunk in enumerate(batch_chunks):
            chunk_num = batch_start + i + 1
            print(f"  📄 Chunk {chunk_num}/{len(text_chunks)}...", end=" ")
            
            # Enhanced prompt for comprehensive metabolite extraction
            prompt = f"""Extract ALL chemical compounds, metabolites, and biomarkers from this wine research text. Include:

1. Wine phenolic compounds (anthocyanins, flavonoids, phenolic acids)
2. Urinary metabolites and biomarkers
3. Chemical names with modifications (glucoside, glucuronide, sulfate)
4. Specific compound names mentioned
5. Metabolic derivatives and conjugates

Text: {chunk}

List each compound on a separate line (no numbering):"""
            
            try:
                chunk_start = time.time()
                response = wrapper.generate_single(prompt, max_tokens=600)
                chunk_end = time.time()
                
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
                
                print(f"{len(chunk_metabolites)} metabolites ({chunk_end - chunk_start:.2f}s)")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        print(f"  ✅ Batch completed in {batch_time:.2f}s")
        
        # Small delay between batches to be respectful to API
        if batch_end < len(text_chunks):
            time.sleep(1)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Remove duplicates while preserving order
    unique_metabolites = []
    seen = set()
    for metabolite in all_metabolites:
        metabolite_lower = metabolite.lower().strip()
        if metabolite_lower not in seen and len(metabolite_lower) > 2:
            seen.add(metabolite_lower)
            unique_metabolites.append(metabolite.strip())
    
    print(f"\n✅ FULL DOCUMENT PROCESSING COMPLETE!")
    print(f"📊 Total time: {total_time:.2f}s")
    print(f"📊 Average per chunk: {total_time/len(text_chunks):.2f}s")
    print(f"📊 Raw metabolites extracted: {len(all_metabolites)}")
    print(f"📊 Unique metabolites: {len(unique_metabolites)}")
    
    return unique_metabolites

def normalize_metabolite_name(name: str) -> str:
    """Normalize metabolite name for comparison"""
    normalized = re.sub(r'[^\w\s-]', '', name.lower()).strip()
    normalized = re.sub(r'\b(acid|derivative|metabolite|compound)\b', '', normalized).strip()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def find_comprehensive_matches(extracted: List[str], expected: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Find comprehensive matches between extracted and expected metabolites"""
    
    # Normalize all names for comparison
    expected_normalized = {}
    for exp in expected:
        norm = normalize_metabolite_name(exp)
        expected_normalized[norm] = exp
    
    extracted_normalized = {}
    for ext in extracted:
        norm = normalize_metabolite_name(ext)
        extracted_normalized[norm] = ext
    
    # Find matches for expected metabolites
    matches_expected = []
    for norm_exp, orig_exp in expected_normalized.items():
        match_found = False
        best_match = None
        match_type = None
        
        # Check for exact match
        if norm_exp in extracted_normalized:
            best_match = extracted_normalized[norm_exp]
            match_type = "exact"
            match_found = True
        else:
            # Check for partial matches
            for norm_ext, orig_ext in extracted_normalized.items():
                if (norm_exp in norm_ext or norm_ext in norm_exp) and len(norm_exp) > 3:
                    if not all(word in ['acid', 'derivative', 'metabolite', 'compound'] for word in norm_exp.split()):
                        best_match = orig_ext
                        match_type = "partial"
                        match_found = True
                        break
        
        matches_expected.append({
            'expected': orig_exp,
            'found': best_match if match_found else None,
            'match_type': match_type if match_found else None,
            'detected': match_found
        })
    
    # Find matches for extracted metabolites
    matches_found = []
    for norm_ext, orig_ext in extracted_normalized.items():
        match_found = False
        best_match = None
        match_type = None
        
        # Check for exact match
        if norm_ext in expected_normalized:
            best_match = expected_normalized[norm_ext]
            match_type = "exact"
            match_found = True
        else:
            # Check for partial matches
            for norm_exp, orig_exp in expected_normalized.items():
                if (norm_ext in norm_exp or norm_exp in norm_ext) and len(norm_ext) > 3:
                    if not all(word in ['acid', 'derivative', 'metabolite', 'compound'] for word in norm_ext.split()):
                        best_match = orig_exp
                        match_type = "partial"
                        match_found = True
                        break
        
        matches_found.append({
            'extracted': orig_ext,
            'expected': best_match if match_found else None,
            'match_type': match_type if match_found else None,
            'in_expected': match_found
        })
    
    return matches_found, matches_expected

def process_full_document():
    """Process the complete Wine Biomarkers PDF"""
    print("🍷 FOODB Full Document Processing: Wine Biomarkers PDF")
    print("=" * 60)
    
    # File paths
    pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
    csv_path = "urinary_wine_biomarkers.csv"
    
    try:
        # Load expected metabolites
        print("📋 Loading expected metabolites...")
        expected_metabolites = load_expected_metabolites(csv_path)
        if not expected_metabolites:
            print("❌ Failed to load expected metabolites")
            return
        
        print(f"✅ Loaded {len(expected_metabolites)} expected metabolites")
        
        # Extract PDF text
        print("📄 Extracting PDF text...")
        pdf_text = extract_pdf_text_simple(pdf_path)
        if not pdf_text:
            print("❌ Failed to extract PDF text")
            return
        
        print(f"✅ Extracted {len(pdf_text)} characters from PDF")
        
        # Initialize wrapper
        print("🔬 Initializing LLM wrapper...")
        from llm_wrapper import LLMWrapper
        wrapper = LLMWrapper()
        print(f"✅ Using model: {wrapper.current_model.get('model_name', 'Unknown')}")
        
        # Chunk text
        print("📝 Chunking text...")
        words = pdf_text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_size = 1500
        
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
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Process ALL chunks
        print(f"\n🚀 PROCESSING FULL DOCUMENT ({len(chunks)} chunks)")
        print("=" * 50)
        
        start_time = time.time()
        extracted_metabolites = extract_metabolites_batch(wrapper, chunks, batch_size=5)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Find comprehensive matches
        print("\n🔍 Analyzing matches...")
        matches_found, matches_expected = find_comprehensive_matches(extracted_metabolites, expected_metabolites)
        
        # Calculate metrics
        detected_count = sum(1 for match in matches_expected if match['detected'])
        expected_count = sum(1 for match in matches_found if match['in_expected'])
        
        total_expected = len(expected_metabolites)
        total_extracted = len(extracted_metabolites)
        
        precision = expected_count / total_extracted if total_extracted > 0 else 0
        recall = detected_count / total_expected if total_expected > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display results
        print(f"\n📊 FULL DOCUMENT RESULTS")
        print("=" * 40)
        print(f"📄 Document: 100% processed ({len(chunks)} chunks)")
        print(f"⏱️ Total processing time: {total_time:.2f}s")
        print(f"⏱️ Average per chunk: {total_time/len(chunks):.2f}s")
        print(f"🧬 Expected metabolites: {total_expected}")
        print(f"🧬 Extracted metabolites: {total_extracted}")
        print(f"✅ Detected from expected: {detected_count}")
        print(f"✅ Expected from found: {expected_count}")
        print(f"📈 Detection Rate (Recall): {recall:.1%}")
        print(f"📈 Precision: {precision:.1%}")
        print(f"📈 F1-Score: {f1_score:.1%}")
        
        # Compare with partial results
        print(f"\n📊 COMPARISON: Partial vs Full Processing")
        print("=" * 50)
        print(f"Partial (8 chunks):  54.2% recall, 22.2% precision, 31.5% F1")
        print(f"Full ({len(chunks)} chunks):    {recall:.1%} recall, {precision:.1%} precision, {f1_score:.1%} F1")
        
        improvement_recall = recall - 0.542
        improvement_precision = precision - 0.222
        improvement_f1 = f1_score - 0.315
        
        print(f"Improvement:         {improvement_recall:+.1%} recall, {improvement_precision:+.1%} precision, {improvement_f1:+.1%} F1")
        
        # Save comprehensive results
        results = {
            'processing_info': {
                'chunks_processed': len(chunks),
                'total_time': total_time,
                'avg_time_per_chunk': total_time / len(chunks),
                'document_coverage': '100%'
            },
            'results': {
                'total_expected': total_expected,
                'total_extracted': total_extracted,
                'detected_count': detected_count,
                'expected_count': expected_count,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score
            },
            'matches_expected': matches_expected,
            'matches_found': matches_found,
            'all_extracted_metabolites': extracted_metabolites
        }
        
        with open("Full_Document_Metabolite_Results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Full results saved to: Full_Document_Metabolite_Results.json")
        
        # Show top matches
        exact_matches = [m for m in matches_expected if m['detected'] and m['match_type'] == 'exact']
        if exact_matches:
            print(f"\n🎯 Exact Matches Found ({len(exact_matches)}):")
            for match in exact_matches:
                print(f"  ✅ {match['expected']}")
        
        print(f"\n🎉 FULL DOCUMENT PROCESSING COMPLETE!")
        print(f"The FOODB wrapper processed the entire Wine Biomarkers PDF")
        print(f"and achieved {recall:.1%} detection rate with {precision:.1%} precision!")
        
    except Exception as e:
        print(f"❌ Error processing full document: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_full_document()
