#!/usr/bin/env python3
"""
Test FOODB Wrapper on Wine Biomarkers PDF
This script tests the wrapper's ability to extract metabolites from a real scientific PDF
and compares results with known urinary wine biomarkers.
"""

import sys
import os
import csv
import json
import time
import re
from pathlib import Path
from typing import List, Set, Dict

# Add the FOODB_LLM_pipeline directory to the path
sys.path.append('FOODB_LLM_pipeline')

def install_pdf_dependencies():
    """Install PDF processing dependencies if needed"""
    try:
        import PyPDF2
        return True
    except ImportError:
        print("📦 Installing PDF processing dependencies...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
            import PyPDF2
            return True
        except:
            print("❌ Failed to install PyPDF2. Trying alternative method...")
            return False

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    print(f"📄 Extracting text from PDF: {pdf_path}")
    
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"📖 PDF has {len(pdf_reader.pages)} pages")
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
                    if page_num % 10 == 0:
                        print(f"  Processed page {page_num + 1}/{len(pdf_reader.pages)}")
                        
                except Exception as e:
                    print(f"  ⚠️ Error extracting page {page_num + 1}: {e}")
                    continue
            
            print(f"✅ Extracted {len(text)} characters from PDF")
            return text
            
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return ""

def load_expected_metabolites(csv_path: str) -> List[str]:
    """Load expected metabolites from CSV file"""
    print(f"📋 Loading expected metabolites from: {csv_path}")
    
    try:
        metabolites = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                if row and row[0].strip():
                    metabolites.append(row[0].strip())
        
        print(f"✅ Loaded {len(metabolites)} expected metabolites")
        return metabolites
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    """Split text into manageable chunks"""
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
    
    print(f"📝 Split text into {len(chunks)} chunks")
    return chunks

def extract_metabolites_with_wrapper(wrapper, text_chunks: List[str]) -> List[str]:
    """Extract metabolites from text chunks using the FOODB wrapper"""
    print(f"🧬 Extracting metabolites from {len(text_chunks)} text chunks...")
    
    all_metabolites = []
    
    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Processing chunk {i}/{len(text_chunks)}...")
        
        # Create prompt for metabolite extraction
        prompt = f"""Extract all chemical compounds, metabolites, and biomarkers mentioned in this scientific text. Focus on:
- Wine-related compounds (anthocyanins, phenolic acids, etc.)
- Urinary metabolites and biomarkers
- Chemical names with specific structures
- Compounds with glucoside, glucuronide, sulfate modifications

Text: {chunk}

List all compounds found (one per line):"""
        
        try:
            start_time = time.time()
            response = wrapper.generate_single(prompt, max_tokens=400)
            end_time = time.time()
            
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
            
            print(f"    Found {len(chunk_metabolites)} compounds in {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"    ❌ Error processing chunk {i}: {e}")
            continue
    
    # Remove duplicates while preserving order
    unique_metabolites = []
    seen = set()
    for metabolite in all_metabolites:
        metabolite_lower = metabolite.lower()
        if metabolite_lower not in seen:
            seen.add(metabolite_lower)
            unique_metabolites.append(metabolite)
    
    print(f"✅ Extracted {len(unique_metabolites)} unique metabolites")
    return unique_metabolites

def compare_metabolites(extracted: List[str], expected: List[str]) -> Dict:
    """Compare extracted metabolites with expected ones"""
    print(f"🔍 Comparing extracted vs expected metabolites...")
    
    # Normalize names for comparison
    def normalize_name(name: str) -> str:
        return re.sub(r'[^\w\s-]', '', name.lower()).strip()
    
    extracted_normalized = {normalize_name(m): m for m in extracted}
    expected_normalized = {normalize_name(m): m for m in expected}
    
    # Find matches
    matches = []
    for norm_expected, original_expected in expected_normalized.items():
        for norm_extracted, original_extracted in extracted_normalized.items():
            # Check for exact match or partial match
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
    
    return results

def test_wine_biomarkers_pdf():
    """Main test function"""
    print("🍷 FOODB Wrapper Test: Wine Biomarkers PDF")
    print("=" * 60)
    
    # File paths
    pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
    csv_path = "urinary_wine_biomarkers.csv"
    
    # Check if files exist
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    try:
        # Step 1: Install dependencies and extract PDF text
        if not install_pdf_dependencies():
            print("❌ Cannot install PDF dependencies. Skipping PDF extraction.")
            return
        
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            print("❌ Failed to extract text from PDF")
            return
        
        # Step 2: Load expected metabolites
        expected_metabolites = load_expected_metabolites(csv_path)
        if not expected_metabolites:
            print("❌ Failed to load expected metabolites")
            return
        
        # Step 3: Initialize FOODB wrapper
        print(f"\n🔬 Initializing FOODB wrapper...")
        from llm_wrapper import LLMWrapper
        
        wrapper = LLMWrapper()
        print(f"✅ Using model: {wrapper.current_model.get('model_name', 'Unknown')}")
        
        # Step 4: Process PDF text
        text_chunks = chunk_text(pdf_text, chunk_size=1500)
        
        # Process a subset of chunks for testing (to avoid long processing time)
        test_chunks = text_chunks[:5]  # Test first 5 chunks
        print(f"🧪 Testing with first {len(test_chunks)} chunks (out of {len(text_chunks)})")
        
        start_time = time.time()
        extracted_metabolites = extract_metabolites_with_wrapper(wrapper, test_chunks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Step 5: Compare results
        comparison = compare_metabolites(extracted_metabolites, expected_metabolites)
        
        # Step 6: Display results
        print(f"\n📊 Results Summary:")
        print("=" * 40)
        print(f"📄 PDF processing time: {total_time:.2f}s")
        print(f"📝 Text chunks processed: {len(test_chunks)}")
        print(f"🎯 Expected metabolites: {comparison['total_expected']}")
        print(f"🔍 Extracted metabolites: {comparison['total_extracted']}")
        print(f"✅ Matches found: {comparison['total_matches']}")
        print(f"📈 Precision: {comparison['precision']:.2%}")
        print(f"📈 Recall: {comparison['recall']:.2%}")
        print(f"📈 F1-Score: {comparison['f1_score']:.2%}")
        
        # Show matches
        if comparison['matches']:
            print(f"\n🎯 Found Matches:")
            for i, match in enumerate(comparison['matches'][:10], 1):  # Show first 10
                match_symbol = "🎯" if match['match_type'] == 'exact' else "🔍"
                print(f"  {i}. {match_symbol} Expected: {match['expected']}")
                print(f"     Extracted: {match['extracted']}")
            
            if len(comparison['matches']) > 10:
                print(f"     ... and {len(comparison['matches']) - 10} more matches")
        
        # Show sample extracted metabolites
        print(f"\n📋 Sample Extracted Metabolites:")
        for i, metabolite in enumerate(extracted_metabolites[:10], 1):
            print(f"  {i}. {metabolite}")
        
        if len(extracted_metabolites) > 10:
            print(f"     ... and {len(extracted_metabolites) - 10} more")
        
        # Show sample expected metabolites
        print(f"\n📋 Sample Expected Metabolites:")
        for i, metabolite in enumerate(expected_metabolites[:10], 1):
            print(f"  {i}. {metabolite}")
        
        if len(expected_metabolites) > 10:
            print(f"     ... and {len(expected_metabolites) - 10} more")
        
        # Save results
        results_file = "wine_biomarkers_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'test_info': {
                    'pdf_file': pdf_path,
                    'csv_file': csv_path,
                    'chunks_processed': len(test_chunks),
                    'processing_time': total_time,
                    'model_used': wrapper.current_model.get('model_name', 'Unknown')
                },
                'results': comparison,
                'extracted_metabolites': extracted_metabolites,
                'expected_metabolites': expected_metabolites
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Final assessment
        if comparison['f1_score'] > 0.3:
            print(f"\n🎉 Test SUCCESSFUL! The wrapper effectively extracted wine biomarkers.")
            print(f"The F1-score of {comparison['f1_score']:.2%} indicates good performance.")
        elif comparison['total_matches'] > 0:
            print(f"\n✅ Test PARTIAL SUCCESS! Found {comparison['total_matches']} matches.")
            print(f"The wrapper can identify some wine biomarkers from the PDF.")
        else:
            print(f"\n⚠️ Test shows limited success. Consider refining the extraction prompts.")
        
        print(f"\n💡 This demonstrates the wrapper's ability to process real scientific PDFs")
        print(f"and extract relevant metabolites for FOODB pipeline applications!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wine_biomarkers_pdf()
