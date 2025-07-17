#!/usr/bin/env python3
"""
Comprehensive OpenRouter Models Evaluation
Tests OpenRouter models against Wine-consumptionbiomarkers-HMDB.pdf using urinary_wine_biomarkers.csv as ground truth
Generates F1 scores, precision, recall, and processing times for direct comparison with Groq models
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

# Add FOODB pipeline to path
sys.path.append('FOODB_LLM_pipeline')
from openrouter_client import OpenRouterClient

def load_env_file():
    """Load environment variables from .env file exclusively"""
    env_file = Path('.env')
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    return env_vars

def setup_logging():
    """Setup logging configuration"""
    log_filename = f"comprehensive_openrouter_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return log_filename

def load_ground_truth():
    """Load ground truth biomarkers from CSV"""
    try:
        df = pd.read_csv('urinary_wine_biomarkers.csv')
        # Get unique biomarker names, handling different possible column names
        if 'Biomarker' in df.columns:
            biomarkers = df['Biomarker'].dropna().unique().tolist()
        elif 'biomarker' in df.columns:
            biomarkers = df['biomarker'].dropna().unique().tolist()
        elif 'name' in df.columns:
            biomarkers = df['name'].dropna().unique().tolist()
        else:
            # Use first column if no standard name found
            biomarkers = df.iloc[:, 0].dropna().unique().tolist()
        
        # Clean biomarker names
        biomarkers = [str(b).strip().lower() for b in biomarkers if str(b).strip()]
        logging.info(f"Loaded {len(biomarkers)} ground truth biomarkers")
        return biomarkers
    except Exception as e:
        logging.error(f"Error loading ground truth: {e}")
        return []

def get_openrouter_models():
    """Get list of working OpenRouter models from V3 file"""
    try:
        with open('free_models_reasoning_ranked_v3.json', 'r') as f:
            models_data = json.load(f)
        
        # Extract model IDs from V3 file
        models = []
        for model in models_data:
            if model.get('provider') == 'OpenRouter':
                models.append(model['model_id'])
        
        logging.info(f"Found {len(models)} OpenRouter models in V3 file")
        return models
    except Exception as e:
        logging.error(f"Error loading OpenRouter models: {e}")
        # Fallback to known working models
        return [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free", 
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "mistralai/mistral-nemo:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-3-12b-it:free",
            "mistralai/mistral-small-3.2-24b-instruct:free",
            "deepseek/deepseek-chat:free",
            "google/gemma-3-27b-it:free",
            "moonshotai/kimi-vl-a3b-thinking:free",
            "deepseek/deepseek-v3-base:free",
            "qwen/qwen3-235b-a22b:free",
            "tngtech/deepseek-r1t-chimera:free",
            "nousresearch/deephermes-3-llama-3-8b-preview:free",
            "moonshotai/kimi-k2:free"
        ]

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
            logging.error("Neither PyPDF2 nor PyMuPDF available for PDF extraction")
            return ""
    except Exception as e:
        logging.error(f"Error extracting PDF text: {e}")
        return ""

def create_metabolite_extraction_prompt(text_chunk):
    """Create prompt for metabolite extraction"""
    return f"""You are a scientific expert specializing in metabolomics and biomarker analysis. Your task is to extract ALL metabolites, biomarkers, and chemical compounds mentioned in the following scientific text.

Please identify and list:
1. All metabolites and biomarkers
2. Chemical compounds and their derivatives
3. Metabolic products and intermediates
4. Any molecules that could serve as biomarkers

For each compound found, provide just the name. Be comprehensive and include all variants, derivatives, and related compounds mentioned.

Text to analyze:
{text_chunk}

Please provide a JSON list of compound names:
["compound1", "compound2", "compound3", ...]

Only return the JSON list, nothing else."""

def extract_metabolites_from_response(response):
    """Extract metabolite names from LLM response"""
    metabolites = []

    # Try to parse as JSON first
    try:
        # Look for JSON array in response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            metabolites = json.loads(json_str)
            return [str(m).strip() for m in metabolites if str(m).strip()]
    except:
        pass

    # Fallback: extract from text
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('*'):
            # Remove common prefixes
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^-\s*', '', line)
            line = re.sub(r'^\*\s*', '', line)
            if line:
                metabolites.append(line)

    return metabolites

def calculate_metrics(extracted_metabolites, ground_truth):
    """Calculate precision, recall, and F1 score"""
    if not extracted_metabolites:
        return 0.0, 0.0, 0.0, 0, 0

    # Convert to lowercase for comparison
    extracted_lower = [m.lower().strip() for m in extracted_metabolites]
    ground_truth_lower = [b.lower().strip() for b in ground_truth]

    # Find matches using fuzzy matching
    true_positives = 0
    matched_biomarkers = []

    for metabolite in extracted_lower:
        for biomarker in ground_truth_lower:
            # Check for exact match or substring match
            if (metabolite == biomarker or
                metabolite in biomarker or
                biomarker in metabolite or
                # Check for common variations
                metabolite.replace('-', '') == biomarker.replace('-', '') or
                metabolite.replace(' ', '') == biomarker.replace(' ', '')):
                if biomarker not in matched_biomarkers:
                    true_positives += 1
                    matched_biomarkers.append(biomarker)
                break

    false_positives = len(extracted_metabolites) - true_positives
    false_negatives = len(ground_truth) - true_positives

    # Calculate metrics
    precision = true_positives / len(extracted_metabolites) if extracted_metabolites else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score, true_positives, len(extracted_metabolites)

def evaluate_model(model_id, ground_truth, env_vars, pdf_text):
    """Evaluate a single OpenRouter model directly"""
    logging.info(f"Evaluating model: {model_id}")

    try:
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value

        # Initialize OpenRouter client with specific model
        client = OpenRouterClient(api_key=env_vars.get('OPENROUTER_API_KEY'), model=model_id)

        # Split text into chunks (similar to pipeline approach)
        chunk_size = 2000
        chunks = []
        words = pdf_text.split()

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        logging.info(f"Processing {len(chunks)} chunks for model {model_id}")

        # Process each chunk
        all_metabolites = []
        start_time = time.time()

        for i, chunk in enumerate(chunks):
            try:
                prompt = create_metabolite_extraction_prompt(chunk)
                response = client.generate_single(prompt, max_tokens=500, temperature=0.1)

                if response:
                    metabolites = extract_metabolites_from_response(response)
                    all_metabolites.extend(metabolites)
                    logging.debug(f"Chunk {i+1}/{len(chunks)}: Found {len(metabolites)} metabolites")

                # Add small delay to avoid rate limiting
                time.sleep(1)

            except Exception as e:
                logging.warning(f"Error processing chunk {i+1} for model {model_id}: {e}")
                continue

        processing_time = time.time() - start_time

        # Remove duplicates and clean up
        unique_metabolites = []
        seen = set()
        for metabolite in all_metabolites:
            clean_metabolite = metabolite.strip().lower()
            if clean_metabolite and clean_metabolite not in seen:
                unique_metabolites.append(metabolite.strip())
                seen.add(clean_metabolite)

        # Calculate metrics
        precision, recall, f1_score, biomarkers_detected, total_extracted = calculate_metrics(
            unique_metabolites, ground_truth
        )

        result_data = {
            "model_id": model_id,
            "success": True,
            "processing_time": processing_time,
            "chunks_processed": len(chunks),
            "metabolites_extracted": total_extracted,
            "biomarkers_detected": biomarkers_detected,
            "total_ground_truth": len(ground_truth),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "extracted_metabolites": unique_metabolites[:30],  # First 30 for analysis
            "error": None
        }

        logging.info(f"Model {model_id}: F1={f1_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Time={processing_time:.1f}s, Chunks={len(chunks)}")
        return result_data

    except Exception as e:
        logging.error(f"Error evaluating model {model_id}: {e}")
        return {
            "model_id": model_id,
            "success": False,
            "processing_time": 0,
            "chunks_processed": 0,
            "metabolites_extracted": 0,
            "biomarkers_detected": 0,
            "total_ground_truth": len(ground_truth),
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "extracted_metabolites": [],
            "error": str(e)
        }

def main():
    """Main evaluation function"""
    print("🧪 Starting Comprehensive OpenRouter Models Evaluation")
    print("=" * 60)
    
    # Setup logging
    log_filename = setup_logging()
    logging.info("Starting comprehensive OpenRouter evaluation")
    
    # Load environment variables from .env file
    env_vars = load_env_file()
    if 'OPENROUTER_API_KEY' not in env_vars:
        logging.error("OPENROUTER_API_KEY not found in .env file")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth()
    if not ground_truth:
        logging.error("Failed to load ground truth biomarkers")
        return
    
    # Get OpenRouter models
    models = get_openrouter_models()
    if not models:
        logging.error("No OpenRouter models found")
        return

    # Extract PDF text
    pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return

    print(f"📄 Extracting text from {pdf_path}...")
    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text:
        logging.error("Failed to extract text from PDF")
        return

    print(f"✅ Extracted {len(pdf_text)} characters from PDF")
    print(f"📊 Evaluating {len(models)} OpenRouter models")
    print(f"📄 Document: Wine-consumptionbiomarkers-HMDB.pdf")
    print(f"🎯 Ground Truth: {len(ground_truth)} biomarkers from urinary_wine_biomarkers.csv")
    print()

    # Evaluate each model
    results = []
    for i, model_id in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Testing {model_id}...")
        result = evaluate_model(model_id, ground_truth, env_vars, pdf_text)
        results.append(result)

        # Add delay between requests to avoid rate limiting
        if i < len(models):
            time.sleep(5)
    
    # Sort results by F1 score
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"comprehensive_openrouter_evaluation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    successful_results = [r for r in results if r['success']]
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE OPENROUTER EVALUATION RESULTS")
    print("=" * 60)
    print(f"✅ Successful evaluations: {len(successful_results)}/{len(models)}")
    print(f"📄 Document: Wine-consumptionbiomarkers-HMDB.pdf")
    print(f"🎯 Ground Truth: {len(ground_truth)} biomarkers")
    print()
    
    if successful_results:
        print("🏆 PERFORMANCE RANKING BY F1 SCORE:")
        print("-" * 40)
        for i, result in enumerate(successful_results, 1):
            print(f"{i:2d}. {result['model_id']}")
            print(f"    F1: {result['f1_score']:.4f} | Precision: {result['precision']:.4f} | Recall: {result['recall']:.4f}")
            print(f"    Time: {result['processing_time']:.1f}s | Chunks: {result.get('chunks_processed', 0)} | Detected: {result['biomarkers_detected']}/{len(ground_truth)} biomarkers")
            print()
        
        # Calculate averages
        avg_f1 = sum(r['f1_score'] for r in successful_results) / len(successful_results)
        avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        
        print("📈 AVERAGE PERFORMANCE:")
        print(f"   F1 Score: {avg_f1:.4f}")
        print(f"   Precision: {avg_precision:.4f}")
        print(f"   Recall: {avg_recall:.4f}")
        print(f"   Processing Time: {avg_time:.1f}s")
    
    # Show failed models
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\n❌ FAILED MODELS:")
        for result in failed_results:
            print(f"   {result['model_id']}: {result['error']}")
    
    print(f"\n📋 Detailed results saved to: {results_file}")
    print(f"📋 Log file: {log_filename}")
    print("\n🎉 Comprehensive OpenRouter evaluation completed!")

if __name__ == "__main__":
    main()
