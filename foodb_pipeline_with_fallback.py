#!/usr/bin/env python3
"""
FOODB Pipeline with Enhanced Fallback Functionality
This script demonstrates how to integrate the enhanced wrapper with fallback
capabilities into the existing FOODB pipeline for production use.
"""

import sys
import os
import csv
import json
import time
import re
from pathlib import Path
from typing import List, Dict
from enhanced_llm_wrapper_with_fallback import EnhancedLLMWrapper, RetryConfig

def setup_enhanced_pipeline(max_attempts: int = 3, base_delay: float = 1.0):
    """Setup enhanced pipeline with fallback configuration"""
    print("🔧 Setting up Enhanced FOODB Pipeline with Fallback")
    print("=" * 60)
    
    # Configure retry behavior
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=30.0,  # Max 30 seconds delay
        exponential_base=2.0,
        jitter=True
    )
    
    # Create enhanced wrapper
    wrapper = EnhancedLLMWrapper(retry_config=retry_config)
    
    # Show configuration
    print(f"📋 Fallback Configuration:")
    print(f"  Max Retry Attempts: {retry_config.max_attempts}")
    print(f"  Base Delay: {retry_config.base_delay}s")
    print(f"  Max Delay: {retry_config.max_delay}s")
    print(f"  Exponential Base: {retry_config.exponential_base}")
    print(f"  Jitter Enabled: {retry_config.jitter}")
    
    # Show provider status
    print(f"\n🌐 Provider Status:")
    status = wrapper.get_provider_status()
    print(f"  Primary Provider: {status['current_provider']}")
    
    for provider, info in status['providers'].items():
        api_status = "✅ Ready" if info['has_api_key'] else "❌ No API Key"
        print(f"  {provider.title()}: {info['status']} ({api_status})")
    
    return wrapper

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

def process_chunks_with_fallback(wrapper: EnhancedLLMWrapper, text_chunks: List[str], 
                                batch_size: int = 3) -> List[str]:
    """Process text chunks with enhanced fallback capabilities"""
    print(f"\n🧬 Processing {len(text_chunks)} chunks with fallback protection")
    print("=" * 60)
    
    all_metabolites = []
    
    # Process in smaller batches to reduce rate limiting
    for batch_start in range(0, len(text_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(text_chunks))
        batch_chunks = text_chunks[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        
        print(f"\n📦 Batch {batch_num}/{(len(text_chunks)-1)//batch_size + 1} "
              f"(chunks {batch_start+1}-{batch_end})")
        
        # Show current provider status
        status = wrapper.get_provider_status()
        print(f"  Current Provider: {status['current_provider']}")
        
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
            
            chunk_start_time = time.time()
            
            # Use enhanced wrapper with fallback
            response = wrapper.generate_single_with_fallback(prompt, max_tokens=600)
            
            chunk_end_time = time.time()
            chunk_time = chunk_end_time - chunk_start_time
            
            # Parse metabolites from response
            chunk_metabolites = []
            if response:
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
                        'biomarkers', 'chemical', 'wine', 'urinary', 'list', 'include',
                        'based on', 'from the text', 'mentioned in'
                    ]
                    
                    if (line and len(line) > 3 and 
                        not any(pattern in line.lower() for pattern in skip_patterns) and
                        not line.startswith(('Here', 'The', 'List', 'Compounds', 'Based on', 'From the'))):
                        chunk_metabolites.append(line)
            
            all_metabolites.extend(chunk_metabolites)
            
            # Show result with provider info
            current_provider = wrapper.get_provider_status()['current_provider']
            print(f"{len(chunk_metabolites)} metabolites ({chunk_time:.2f}s) [{current_provider}]")
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        print(f"  ✅ Batch completed in {batch_time:.2f}s")
        
        # Show statistics after each batch
        stats = wrapper.get_statistics()
        if stats['fallback_switches'] > 0 or stats['rate_limited_requests'] > 0:
            print(f"  📊 Fallbacks: {stats['fallback_switches']}, Rate Limits: {stats['rate_limited_requests']}")
        
        # Longer delay between batches to be respectful to APIs
        if batch_end < len(text_chunks):
            print(f"  ⏱️ Waiting 2s between batches...")
            time.sleep(2)
    
    # Remove duplicates
    unique_metabolites = []
    seen = set()
    for metabolite in all_metabolites:
        metabolite_lower = metabolite.lower().strip()
        if metabolite_lower not in seen and len(metabolite_lower) > 2:
            seen.add(metabolite_lower)
            unique_metabolites.append(metabolite.strip())
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Raw metabolites: {len(all_metabolites)}")
    print(f"📊 Unique metabolites: {len(unique_metabolites)}")
    
    return unique_metabolites

def run_enhanced_pipeline_demo():
    """Run demonstration of enhanced pipeline with fallback"""
    print("🚀 FOODB Enhanced Pipeline with Fallback Demo")
    print("=" * 60)
    
    # File paths
    pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
    csv_path = "urinary_wine_biomarkers.csv"
    
    try:
        # Setup enhanced pipeline
        wrapper = setup_enhanced_pipeline(max_attempts=3, base_delay=1.0)
        
        # Extract PDF text
        print(f"\n📄 Extracting PDF text...")
        pdf_text = extract_pdf_text_simple(pdf_path)
        if not pdf_text:
            print("❌ Failed to extract PDF text")
            return
        print(f"✅ Extracted {len(pdf_text)} characters")
        
        # Load expected metabolites
        print(f"\n📋 Loading expected metabolites...")
        expected_metabolites = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header
                
                for row in csv_reader:
                    if row and row[0].strip():
                        expected_metabolites.append(row[0].strip())
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return
        
        print(f"✅ Loaded {len(expected_metabolites)} expected metabolites")
        
        # Chunk text
        print(f"\n📝 Chunking text...")
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
        
        # Process first 10 chunks for demo (to avoid long processing time)
        demo_chunks = chunks[:10]
        print(f"\n🧪 Demo: Processing first {len(demo_chunks)} chunks")
        
        start_time = time.time()
        extracted_metabolites = process_chunks_with_fallback(wrapper, demo_chunks, batch_size=3)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Show final statistics
        print(f"\n📊 Final Results:")
        print(f"  Processing Time: {total_time:.2f}s")
        print(f"  Chunks Processed: {len(demo_chunks)}")
        print(f"  Metabolites Extracted: {len(extracted_metabolites)}")
        print(f"  Average per Chunk: {len(extracted_metabolites)/len(demo_chunks):.1f}")
        
        # Show wrapper statistics
        stats = wrapper.get_statistics()
        print(f"\n📈 Wrapper Statistics:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Rate Limited: {stats['rate_limited_requests']}")
        print(f"  Fallback Switches: {stats['fallback_switches']}")
        print(f"  Retry Attempts: {stats['retry_attempts']}")
        
        # Show provider health
        print(f"\n🏥 Provider Health:")
        status = wrapper.get_provider_status()
        for provider, info in status['providers'].items():
            print(f"  {provider.title()}: {info['status']} (failures: {info['consecutive_failures']})")
        
        # Show sample results
        print(f"\n📋 Sample Extracted Metabolites:")
        for i, metabolite in enumerate(extracted_metabolites[:10], 1):
            print(f"  {i}. {metabolite}")
        
        if len(extracted_metabolites) > 10:
            print(f"  ... and {len(extracted_metabolites) - 10} more")
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = f"enhanced_pipeline_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'processing_time': total_time,
            'chunks_processed': len(demo_chunks),
            'metabolites_extracted': extracted_metabolites,
            'wrapper_statistics': stats,
            'provider_status': status
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        print(f"\n🎉 Enhanced pipeline demo completed successfully!")
        print(f"The enhanced wrapper provides:")
        print(f"  ✅ Automatic fallback between providers")
        print(f"  ✅ Exponential backoff on rate limits")
        print(f"  ✅ Configurable retry attempts")
        print(f"  ✅ Provider health monitoring")
        print(f"  ✅ Comprehensive error handling")
        print(f"  ✅ Production-ready resilience")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_enhanced_pipeline_demo()
