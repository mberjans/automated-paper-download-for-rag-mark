#!/usr/bin/env python3
"""
Check if the chunking in the OpenRouter evaluation covered the entire PDF document
"""

import sys
from pathlib import Path

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except ImportError:
            print("Neither PyPDF2 nor PyMuPDF available for PDF extraction")
            return ""
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def analyze_chunking_coverage():
    """Analyze if the chunking covered the entire document"""
    
    # Extract PDF text (same as in evaluation)
    pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
    if not Path(pdf_path).exists():
        print(f"PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Extracting text from {pdf_path}...")
    pdf_text = extract_pdf_text(pdf_path)
    
    if not pdf_text:
        print("❌ Failed to extract text from PDF")
        return
    
    # Analyze the text
    total_chars = len(pdf_text)
    words = pdf_text.split()
    total_words = len(words)
    
    print(f"📊 DOCUMENT ANALYSIS:")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Total words: {total_words:,}")
    print(f"   Average word length: {total_chars/total_words:.1f} chars/word")
    print()
    
    # Simulate the chunking process (same as in evaluation)
    chunk_size = 2000
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    chunks_created = len(chunks)
    words_in_chunks = sum(len(chunk.split()) for chunk in chunks)
    
    print(f"🔍 CHUNKING ANALYSIS:")
    print(f"   Chunk size: {chunk_size:,} words")
    print(f"   Chunks created: {chunks_created}")
    print(f"   Words in chunks: {words_in_chunks:,}")
    print(f"   Words missed: {total_words - words_in_chunks:,}")
    print(f"   Coverage: {(words_in_chunks/total_words)*100:.1f}%")
    print()
    
    # Analyze each chunk
    print(f"📋 CHUNK BREAKDOWN:")
    for i, chunk in enumerate(chunks, 1):
        chunk_words = len(chunk.split())
        chunk_chars = len(chunk)
        print(f"   Chunk {i}: {chunk_words:,} words, {chunk_chars:,} chars")
    
    print()
    
    # Check if coverage is complete
    if words_in_chunks == total_words:
        print("✅ COMPLETE COVERAGE: All words were processed in chunks")
    else:
        missed_words = total_words - words_in_chunks
        missed_percentage = (missed_words / total_words) * 100
        print(f"❌ INCOMPLETE COVERAGE: {missed_words:,} words ({missed_percentage:.1f}%) were NOT processed")
        
        # Show what was missed
        if missed_words > 0:
            missed_text = ' '.join(words[words_in_chunks:words_in_chunks + min(50, missed_words)])
            print(f"   First 50 missed words: {missed_text}...")
    
    print()
    
    # Expected chunks for complete coverage
    expected_chunks = (total_words + chunk_size - 1) // chunk_size  # Ceiling division
    print(f"📈 EXPECTED vs ACTUAL:")
    print(f"   Expected chunks for complete coverage: {expected_chunks}")
    print(f"   Actual chunks created: {chunks_created}")
    
    if expected_chunks > chunks_created:
        print(f"   ⚠️  Missing {expected_chunks - chunks_created} chunks!")
    else:
        print(f"   ✅ Chunk count matches expectation")

if __name__ == "__main__":
    analyze_chunking_coverage()
