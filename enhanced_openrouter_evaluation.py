#!/usr/bin/env python3
"""
Enhanced OpenRouter Models Evaluation with Exponential Backoff
Includes proper rate limiting handling with wait period doubling
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
import random

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
    log_filename = f"enhanced_openrouter_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return log_filename

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
            "meta-llama/llama-3.3-70b-instruct:free"
        ]

def get_working_openrouter_models():
    """Get comprehensive list of all working OpenRouter models (no failures, no empty responses)"""
    # Based on previous testing, these are all the models that:
    # 1. Successfully connected (no 404, 503 errors)
    # 2. Returned valid responses (no empty responses)
    # 3. Had reasonable performance (detection rate >= 77.8%)

    working_models = [
        # Top performers (100% metabolite detection in basic testing)
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

        # Good performers (77.8% detection in basic testing)
        "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "moonshotai/kimi-k2:free"
    ]

    logging.info(f"Selected {len(working_models)} working OpenRouter models for comprehensive evaluation")
    return working_models

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

def api_call_with_exponential_backoff(client, prompt, max_tokens=500, temperature=0.1, max_retries=5):
    """Make API call with exponential backoff for rate limiting"""
    base_delay = 1.0  # Start with 1 second
    max_delay = 60.0  # Maximum delay of 60 seconds
    
    for attempt in range(max_retries):
        try:
            response = client.generate_single(prompt, max_tokens=max_tokens, temperature=temperature)
            if response:  # Successful response
                return response
            else:
                logging.warning(f"Empty response on attempt {attempt + 1}")
                
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limiting error
            if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                if attempt < max_retries - 1:  # Don't delay on last attempt
                    # Calculate delay with exponential backoff + jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    total_delay = delay + jitter
                    
                    logging.warning(f"Rate limit hit on attempt {attempt + 1}. Waiting {total_delay:.2f}s before retry...")
                    time.sleep(total_delay)
                    continue
                else:
                    logging.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise e
            else:
                # Non-rate-limiting error, re-raise immediately
                logging.error(f"API error on attempt {attempt + 1}: {e}")
                raise e
    
    logging.error(f"Failed to get response after {max_retries} attempts")
    return ""

def evaluate_model(model_id, ground_truth, env_vars, pdf_text):
    """Evaluate a single OpenRouter model with enhanced rate limiting"""
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
        
        # Process each chunk with enhanced rate limiting
        all_metabolites = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            try:
                prompt = create_metabolite_extraction_prompt(chunk)
                
                # Use enhanced API call with exponential backoff
                response = api_call_with_exponential_backoff(
                    client, prompt, max_tokens=500, temperature=0.1, max_retries=5
                )
                
                if response:
                    metabolites = extract_metabolites_from_response(response)
                    all_metabolites.extend(metabolites)
                    logging.debug(f"Chunk {i+1}/{len(chunks)}: Found {len(metabolites)} metabolites")
                
                # Progressive delay between chunks (longer for models that hit rate limits)
                if i < len(chunks) - 1:  # Don't delay after last chunk
                    # Start with 2 seconds, increase slightly for each chunk
                    chunk_delay = 2.0 + (i * 0.5)  # 2.0s, 2.5s, 3.0s, etc.
                    time.sleep(min(chunk_delay, 10.0))  # Cap at 10 seconds
                
            except Exception as e:
                logging.warning(f"Error processing chunk {i+1} for model {model_id}: {e}")
                # Add extra delay after errors
                time.sleep(5.0)
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
            "rate_limiting_strategy": "exponential_backoff_with_jitter",
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
            "rate_limiting_strategy": "exponential_backoff_with_jitter",
            "error": str(e)
        }

def main():
    """Main evaluation function with enhanced rate limiting"""
    print("🧪 Starting Enhanced OpenRouter Models Evaluation with Exponential Backoff")
    print("=" * 70)

    try:
        # Setup logging
        log_filename = setup_logging()
        logging.info("Starting enhanced OpenRouter evaluation with exponential backoff")

        # Load environment variables from .env file
        env_vars = load_env_file()
        if 'OPENROUTER_API_KEY' not in env_vars:
            print("❌ OPENROUTER_API_KEY not found in .env file")
            logging.error("OPENROUTER_API_KEY not found in .env file")
            return

        print("✅ API key loaded from .env file")

        # Load ground truth
        ground_truth = load_ground_truth()
        if not ground_truth:
            print("❌ Failed to load ground truth biomarkers")
            logging.error("Failed to load ground truth biomarkers")
            return

        print(f"✅ Loaded {len(ground_truth)} ground truth biomarkers")

        # Get all working OpenRouter models (no limit for comprehensive testing)
        models = get_working_openrouter_models()  # Test all working models
        if not models:
            print("❌ No OpenRouter models found")
            logging.error("No OpenRouter models found")
            return

        print(f"✅ Found {len(models)} OpenRouter models to test")

        # Extract PDF text
        pdf_path = "Wine-consumptionbiomarkers-HMDB.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            logging.error(f"PDF file not found: {pdf_path}")
            return

        print(f"📄 Extracting text from {pdf_path}...")
        pdf_text = extract_pdf_text(pdf_path)
        if not pdf_text:
            print("❌ Failed to extract text from PDF")
            logging.error("Failed to extract text from PDF")
            return

    except Exception as e:
        print(f"❌ Error in setup: {e}")
        logging.error(f"Error in setup: {e}")
        return
    
    print(f"✅ Extracted {len(pdf_text)} characters from PDF")
    print(f"📊 Evaluating {len(models)} OpenRouter models with enhanced rate limiting")
    print(f"📄 Document: Wine-consumptionbiomarkers-HMDB.pdf")
    print(f"🎯 Ground Truth: {len(ground_truth)} biomarkers from urinary_wine_biomarkers.csv")
    print(f"🔄 Rate Limiting: Exponential backoff with jitter")
    print(f"⏱️  Estimated time: {len(models) * 2:.0f}-{len(models) * 5:.0f} minutes")
    print()

    # Evaluate each model
    results = []
    start_time = time.time()

    for i, model_id in enumerate(models, 1):
        model_start = time.time()
        print(f"[{i}/{len(models)}] Testing {model_id} with enhanced rate limiting...")
        result = evaluate_model(model_id, ground_truth, env_vars, pdf_text)
        results.append(result)

        model_time = time.time() - model_start
        elapsed_total = time.time() - start_time
        avg_time_per_model = elapsed_total / i
        remaining_models = len(models) - i
        estimated_remaining = remaining_models * avg_time_per_model

        print(f"   ✅ Completed in {model_time:.1f}s | F1: {result.get('f1_score', 0):.4f}")
        print(f"   📊 Progress: {i}/{len(models)} | Elapsed: {elapsed_total/60:.1f}min | ETA: {estimated_remaining/60:.1f}min")

        # Progressive delay between models (longer delays for later models to handle rate limits)
        if i < len(models):
            # Adaptive delay based on model performance and rate limiting
            base_delay = 8.0  # Base delay
            progressive_delay = min(i * 1.0, 10.0)  # Progressive increase, capped at 10s
            total_delay = base_delay + progressive_delay

            print(f"   ⏳ Waiting {total_delay:.0f}s before next model...")
            time.sleep(total_delay)
    
    # Sort results by F1 score
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"comprehensive_enhanced_openrouter_evaluation_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate comprehensive summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE ENHANCED OPENROUTER EVALUATION RESULTS")
    print("=" * 80)
    print(f"✅ Successful evaluations: {len(successful_results)}/{len(models)} ({len(successful_results)/len(models)*100:.1f}%)")
    print(f"📄 Document: Wine-consumptionbiomarkers-HMDB.pdf")
    print(f"🎯 Ground Truth: {len(ground_truth)} biomarkers")
    print(f"🔄 Rate Limiting: Exponential backoff with jitter")
    print(f"⏱️  Total evaluation time: {total_time/60:.1f} minutes")
    print()

    if successful_results:
        print("🏆 PERFORMANCE RANKING BY F1 SCORE:")
        print("-" * 80)
        for i, result in enumerate(successful_results, 1):
            model_name = result['model_id'].split('/')[-1].replace(':free', '')
            print(f"{i:2d}. {result['model_id']}")
            print(f"    📊 F1: {result['f1_score']:.4f} | Precision: {result['precision']:.4f} | Recall: {result['recall']:.4f}")
            print(f"    ⏱️  Time: {result['processing_time']:.1f}s | Chunks: {result.get('chunks_processed', 0)} | Detected: {result['biomarkers_detected']}/{len(ground_truth)} biomarkers ({result['biomarkers_detected']/len(ground_truth)*100:.1f}%)")
            print(f"    🔬 Metabolites extracted: {result['metabolites_extracted']}")
            print()

        # Calculate comprehensive statistics
        avg_f1 = sum(r['f1_score'] for r in successful_results) / len(successful_results)
        avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)

        best_f1 = max(successful_results, key=lambda x: x['f1_score'])
        best_recall = max(successful_results, key=lambda x: x['recall'])
        fastest = min(successful_results, key=lambda x: x['processing_time'])

        print("📈 COMPREHENSIVE STATISTICS:")
        print(f"   Average F1 Score: {avg_f1:.4f}")
        print(f"   Average Precision: {avg_precision:.4f}")
        print(f"   Average Recall: {avg_recall:.4f}")
        print(f"   Average Processing Time: {avg_time:.1f}s")
        print()
        print("🏆 TOP PERFORMERS:")
        print(f"   🥇 Best F1 Score: {best_f1['model_id']} (F1: {best_f1['f1_score']:.4f})")
        print(f"   🎯 Best Recall: {best_recall['model_id']} (Recall: {best_recall['recall']:.4f})")
        print(f"   ⚡ Fastest: {fastest['model_id']} (Time: {fastest['processing_time']:.1f}s)")
        print()

        # Performance categories
        excellent_models = [r for r in successful_results if r['f1_score'] >= 0.5]
        good_models = [r for r in successful_results if 0.3 <= r['f1_score'] < 0.5]
        moderate_models = [r for r in successful_results if 0.2 <= r['f1_score'] < 0.3]
        poor_models = [r for r in successful_results if r['f1_score'] < 0.2]

        print("📊 PERFORMANCE CATEGORIES:")
        print(f"   🏆 Excellent (F1 ≥ 0.5): {len(excellent_models)} models")
        print(f"   🥈 Good (0.3 ≤ F1 < 0.5): {len(good_models)} models")
        print(f"   🥉 Moderate (0.2 ≤ F1 < 0.3): {len(moderate_models)} models")
        print(f"   ❌ Poor (F1 < 0.2): {len(poor_models)} models")

    # Show failed models
    if failed_results:
        print(f"\n❌ FAILED MODELS ({len(failed_results)}):")
        for result in failed_results:
            print(f"   {result['model_id']}: {result['error']}")

    print(f"\n📋 Detailed results saved to: {results_file}")
    print(f"📋 Log file: {log_filename}")
    print(f"\n🎉 Comprehensive enhanced OpenRouter evaluation completed!")
    print(f"📊 {len(successful_results)} working models evaluated with exponential backoff rate limiting")

if __name__ == "__main__":
    main()
