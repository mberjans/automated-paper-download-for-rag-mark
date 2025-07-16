#!/usr/bin/env python3
"""
Create detailed metabolite comparison tables
Shows which expected metabolites were detected and which found metabolites were expected
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

def extract_metabolites_detailed(wrapper, text_chunks: List[str]) -> List[str]:
    """Extract metabolites with detailed output"""
    print(f"🧬 Extracting metabolites from {len(text_chunks)} chunks...")
    
    all_metabolites = []
    
    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Processing chunk {i}/{len(text_chunks)}...")
        
        # Enhanced prompt for better metabolite extraction
        prompt = f"""Extract ALL chemical compounds, metabolites, and biomarkers from this scientific text about wine consumption. Include:

1. Wine phenolic compounds (anthocyanins, flavonoids, phenolic acids)
2. Urinary metabolites and biomarkers
3. Chemical names with modifications (glucoside, glucuronide, sulfate)
4. Specific compound names mentioned in the text

Text: {chunk}

List each compound on a separate line (no numbering, no bullets):"""
        
        try:
            response = wrapper.generate_single(prompt, max_tokens=500)
            
            # Parse metabolites from response
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                # Clean up the line
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                line = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
                line = re.sub(r'^[A-Z]\.\s*', '', line)  # Remove letter numbering
                
                # Skip headers and non-compound lines
                skip_patterns = [
                    'here are', 'the following', 'compounds found', 'metabolites',
                    'biomarkers', 'chemical', 'wine', 'urinary', 'list', 'include'
                ]
                
                if (line and len(line) > 3 and 
                    not any(pattern in line.lower() for pattern in skip_patterns) and
                    not line.startswith(('Here', 'The', 'List', 'Compounds', 'Based on'))):
                    all_metabolites.append(line)
            
        except Exception as e:
            print(f"    ❌ Error processing chunk {i}: {e}")
            continue
    
    # Remove duplicates while preserving order
    unique_metabolites = []
    seen = set()
    for metabolite in all_metabolites:
        metabolite_lower = metabolite.lower().strip()
        if metabolite_lower not in seen and len(metabolite_lower) > 2:
            seen.add(metabolite_lower)
            unique_metabolites.append(metabolite.strip())
    
    print(f"✅ Extracted {len(unique_metabolites)} unique metabolites")
    return unique_metabolites

def normalize_metabolite_name(name: str) -> str:
    """Normalize metabolite name for comparison"""
    # Convert to lowercase and remove special characters
    normalized = re.sub(r'[^\w\s-]', '', name.lower()).strip()
    # Remove common prefixes/suffixes that might vary
    normalized = re.sub(r'\b(acid|derivative|metabolite|compound)\b', '', normalized).strip()
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def find_matches(extracted: List[str], expected: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Find matches between extracted and expected metabolites"""
    
    # Normalize all names for comparison
    expected_normalized = {}
    for exp in expected:
        norm = normalize_metabolite_name(exp)
        expected_normalized[norm] = exp
    
    extracted_normalized = {}
    for ext in extracted:
        norm = normalize_metabolite_name(ext)
        extracted_normalized[norm] = ext
    
    # Find matches
    matches_found = []
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
                    # Check if it's a meaningful match (not just common words)
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
    
    # Now check which extracted metabolites match expected ones
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

def create_comparison_tables():
    """Create detailed comparison tables"""
    print("📊 Creating Metabolite Comparison Tables")
    print("=" * 50)
    
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
        
        # Initialize wrapper
        print("🔬 Initializing LLM wrapper...")
        from llm_wrapper import LLMWrapper
        wrapper = LLMWrapper()
        
        # Chunk text
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
        
        # Process first 8 chunks for more comprehensive results
        test_chunks = chunks[:8]
        print(f"📝 Processing {len(test_chunks)} chunks for comprehensive analysis...")
        
        # Extract metabolites
        extracted_metabolites = extract_metabolites_detailed(wrapper, test_chunks)
        
        # Find matches
        print("🔍 Finding matches between extracted and expected metabolites...")
        matches_found, matches_expected = find_matches(extracted_metabolites, expected_metabolites)
        
        # Create tables
        create_expected_metabolites_table(matches_expected)
        create_found_metabolites_table(matches_found)
        create_summary_statistics(matches_expected, matches_found, expected_metabolites, extracted_metabolites)
        
        # Save detailed results
        save_detailed_results(matches_expected, matches_found, expected_metabolites, extracted_metabolites)
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()

def create_expected_metabolites_table(matches_expected: List[Dict]):
    """Create table of expected metabolites with detection status"""
    print(f"\n📋 TABLE 1: EXPECTED METABOLITES (Detection Status)")
    print("=" * 80)
    
    # Count detections
    detected_count = sum(1 for match in matches_expected if match['detected'])
    total_count = len(matches_expected)
    
    print(f"Total Expected: {total_count} | Detected: {detected_count} | Detection Rate: {detected_count/total_count:.1%}")
    print()
    
    # Table header
    print("No.  Expected Metabolite                    Status    Found As")
    print("-" * 80)
    
    # Sort by detection status (detected first)
    sorted_matches = sorted(matches_expected, key=lambda x: (not x['detected'], x['expected']))
    
    for i, match in enumerate(sorted_matches, 1):
        expected = match['expected'][:35].ljust(35)  # Truncate long names
        
        if match['detected']:
            status = "✅ FOUND"
            found_as = match['found'][:25] if match['found'] else ""
            match_type = f"({match['match_type']})" if match['match_type'] else ""
        else:
            status = "❌ MISSING"
            found_as = ""
            match_type = ""
        
        print(f"{i:3d}. {expected} {status:10s} {found_as} {match_type}")
    
    # Save to file
    with open("Expected_Metabolites_Detection_Table.txt", "w") as f:
        f.write("EXPECTED METABOLITES - DETECTION STATUS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Expected: {total_count}\n")
        f.write(f"Detected: {detected_count}\n")
        f.write(f"Detection Rate: {detected_count/total_count:.1%}\n\n")
        
        f.write("No.  Expected Metabolite                    Status      Found As\n")
        f.write("-" * 80 + "\n")
        
        for i, match in enumerate(sorted_matches, 1):
            expected = match['expected'][:35].ljust(35)
            
            if match['detected']:
                status = "FOUND"
                found_as = match['found'] if match['found'] else ""
                match_type = f"({match['match_type']})" if match['match_type'] else ""
            else:
                status = "MISSING"
                found_as = ""
                match_type = ""
            
            f.write(f"{i:3d}. {expected} {status:10s} {found_as} {match_type}\n")
    
    print(f"\n💾 Table saved to: Expected_Metabolites_Detection_Table.txt")

def create_found_metabolites_table(matches_found: List[Dict]):
    """Create table of found metabolites with expected status"""
    print(f"\n📋 TABLE 2: FOUND METABOLITES (Expected Status)")
    print("=" * 80)
    
    # Count expected
    expected_count = sum(1 for match in matches_found if match['in_expected'])
    total_count = len(matches_found)
    
    print(f"Total Found: {total_count} | In Expected List: {expected_count} | Precision: {expected_count/total_count:.1%}")
    print()
    
    # Table header
    print("No.  Found Metabolite                       Status    Expected As")
    print("-" * 80)
    
    # Sort by expected status (expected first)
    sorted_matches = sorted(matches_found, key=lambda x: (not x['in_expected'], x['extracted']))
    
    for i, match in enumerate(sorted_matches, 1):
        extracted = match['extracted'][:35].ljust(35)  # Truncate long names
        
        if match['in_expected']:
            status = "✅ EXPECTED"
            expected_as = match['expected'][:25] if match['expected'] else ""
            match_type = f"({match['match_type']})" if match['match_type'] else ""
        else:
            status = "❓ NEW"
            expected_as = ""
            match_type = ""
        
        print(f"{i:3d}. {extracted} {status:12s} {expected_as} {match_type}")
    
    # Save to file
    with open("Found_Metabolites_Expected_Table.txt", "w") as f:
        f.write("FOUND METABOLITES - EXPECTED STATUS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Found: {total_count}\n")
        f.write(f"In Expected List: {expected_count}\n")
        f.write(f"Precision: {expected_count/total_count:.1%}\n\n")
        
        f.write("No.  Found Metabolite                       Status        Expected As\n")
        f.write("-" * 80 + "\n")
        
        for i, match in enumerate(sorted_matches, 1):
            extracted = match['extracted'][:35].ljust(35)
            
            if match['in_expected']:
                status = "EXPECTED"
                expected_as = match['expected'] if match['expected'] else ""
                match_type = f"({match['match_type']})" if match['match_type'] else ""
            else:
                status = "NEW"
                expected_as = ""
                match_type = ""
            
            f.write(f"{i:3d}. {extracted} {status:12s} {expected_as} {match_type}\n")
    
    print(f"\n💾 Table saved to: Found_Metabolites_Expected_Table.txt")

def create_summary_statistics(matches_expected: List[Dict], matches_found: List[Dict], 
                            expected_metabolites: List[str], extracted_metabolites: List[str]):
    """Create summary statistics"""
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 40)
    
    detected_count = sum(1 for match in matches_expected if match['detected'])
    expected_count = sum(1 for match in matches_found if match['in_expected'])
    
    total_expected = len(expected_metabolites)
    total_extracted = len(extracted_metabolites)
    
    precision = expected_count / total_extracted if total_extracted > 0 else 0
    recall = detected_count / total_expected if total_expected > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Expected Metabolites: {total_expected}")
    print(f"Extracted Metabolites: {total_extracted}")
    print(f"Detected from Expected: {detected_count}")
    print(f"Expected from Found: {expected_count}")
    print(f"Detection Rate (Recall): {recall:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"F1-Score: {f1_score:.1%}")

def save_detailed_results(matches_expected: List[Dict], matches_found: List[Dict], 
                         expected_metabolites: List[str], extracted_metabolites: List[str]):
    """Save detailed results to JSON"""
    results = {
        'summary': {
            'total_expected': len(expected_metabolites),
            'total_extracted': len(extracted_metabolites),
            'detected_count': sum(1 for match in matches_expected if match['detected']),
            'expected_count': sum(1 for match in matches_found if match['in_expected']),
            'detection_rate': sum(1 for match in matches_expected if match['detected']) / len(expected_metabolites),
            'precision': sum(1 for match in matches_found if match['in_expected']) / len(extracted_metabolites)
        },
        'expected_metabolites_analysis': matches_expected,
        'found_metabolites_analysis': matches_found,
        'all_expected_metabolites': expected_metabolites,
        'all_extracted_metabolites': extracted_metabolites
    }
    
    with open("Metabolite_Comparison_Detailed_Results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: Metabolite_Comparison_Detailed_Results.json")

if __name__ == "__main__":
    create_comparison_tables()
