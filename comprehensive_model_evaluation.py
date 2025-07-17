#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for FOODB Pipeline
Evaluates all specified Groq models with F1 scores and timing analysis
"""

import subprocess
import time
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any

def setup_logging():
    """Setup logging for comprehensive evaluation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('comprehensive_model_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def comprehensive_model_evaluation():
    """Comprehensive evaluation of all specified Groq models"""
    logger = setup_logging()
    
    print("🧪 COMPREHENSIVE GROQ MODEL EVALUATION")
    print("=" * 60)
    print("📄 Document: Wine-consumptionbiomarkers-HMDB.pdf")
    print("📊 Ground Truth: urinary_wine_biomarkers.csv")
    print("🎯 Metrics: F1 Score, Processing Time, Accuracy")
    print("=" * 60)
    
    # Models to evaluate (with correct model names)
    models_to_evaluate = [
        "llama-3.3-70b-versatile",
        "moonshotai/kimi-k2-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b"
    ]
    
    # Test parameters
    test_pdf = "Wine-consumptionbiomarkers-HMDB.pdf"
    csv_database = "urinary_wine_biomarkers.csv"
    base_output_dir = "./comprehensive_evaluation"
    
    # Verify files exist
    if not Path(test_pdf).exists():
        logger.error(f"❌ Test PDF not found: {test_pdf}")
        return
    
    if not Path(csv_database).exists():
        logger.error(f"❌ CSV database not found: {csv_database}")
        return
    
    # Create base output directory
    Path(base_output_dir).mkdir(exist_ok=True)
    
    logger.info(f"📋 Evaluating {len(models_to_evaluate)} models")
    logger.info(f"📄 Test document: {test_pdf}")
    logger.info(f"📊 Ground truth: {csv_database}")
    
    results = {}
    
    for i, model in enumerate(models_to_evaluate, 1):
        print(f"\n🔬 Evaluating model {i}/{len(models_to_evaluate)}: {model}")
        print("-" * 70)
        
        try:
            result = evaluate_single_model(
                model, test_pdf, csv_database, base_output_dir, logger
            )
            results[model] = result
            
            if result['success']:
                print(f"✅ {model}: SUCCESS")
                print(f"   ⏱️  Total Time: {result['total_time']:.2f}s")
                print(f"   🧬 Metabolites: {result['metabolites_extracted']}")
                print(f"   🎯 F1 Score: {result['f1_score']:.4f}")
                print(f"   📊 Precision: {result['precision']:.4f}")
                print(f"   📈 Recall: {result['recall']:.4f}")
                print(f"   ✅ Detection Rate: {result['detection_rate']:.1%}")
            else:
                print(f"❌ {model}: FAILED")
                print(f"   Error: {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Exception evaluating {model}: {e}")
            results[model] = {
                'success': False,
                'error': str(e),
                'total_time': 0,
                'f1_score': 0,
                'precision': 0,
                'recall': 0
            }
    
    # Generate comprehensive report
    generate_comprehensive_report(results, logger)
    
    return results

def evaluate_single_model(model: str, test_pdf: str, csv_database: str, base_output_dir: str, logger) -> Dict[str, Any]:
    """Evaluate a single model with comprehensive metrics"""
    
    # Create model-specific output directory
    model_safe_name = model.replace('/', '_').replace('-', '_')
    model_output_dir = f"{base_output_dir}/{model_safe_name}"
    
    start_time = time.time()
    
    try:
        # Run pipeline with specific model
        logger.info(f"Running pipeline with model: {model}")
        
        result = subprocess.run([
            "python", "foodb_pipeline_cli.py",
            test_pdf,
            "--groq-model", model,
            "--output-dir", model_output_dir,
            "--csv-database", csv_database,
            "--document-only",
            "--verify-compounds",
            "--save-timing",
            "--export-format", "json",
            "--chunk-size", "1500",  # Standard chunk size for fair comparison
            "--max-tokens", "200",   # Standard token limit
            "--quiet"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if result.returncode == 0:
            # Analyze results
            analysis = analyze_model_performance(model_output_dir, model, logger)
            analysis['total_time'] = total_time
            analysis['success'] = True
            
            return analysis
        else:
            logger.error(f"Pipeline failed for {model}: {result.stderr}")
            return {
                'success': False,
                'error': result.stderr,
                'total_time': total_time,
                'f1_score': 0,
                'precision': 0,
                'recall': 0,
                'detection_rate': 0,
                'metabolites_extracted': 0
            }
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        total_time = end_time - start_time
        logger.error(f"Timeout for model {model} after {total_time:.1f}s")
        
        return {
            'success': False,
            'error': f'Timeout after {total_time:.1f} seconds',
            'total_time': total_time,
            'f1_score': 0,
            'precision': 0,
            'recall': 0,
            'detection_rate': 0,
            'metabolites_extracted': 0
        }
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        logger.error(f"Exception for model {model}: {e}")
        
        return {
            'success': False,
            'error': str(e),
            'total_time': total_time,
            'f1_score': 0,
            'precision': 0,
            'recall': 0,
            'detection_rate': 0,
            'metabolites_extracted': 0
        }

def analyze_model_performance(output_dir: str, model: str, logger) -> Dict[str, Any]:
    """Analyze model performance from pipeline results"""
    try:
        # Find the results file
        output_path = Path(output_dir)
        json_files = list(output_path.glob("*_results.json"))
        
        if not json_files:
            raise ValueError("No results file found")
        
        # Load the results
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        # Extract comprehensive metrics
        extraction_result = data.get('extraction_result', {})
        matching_result = data.get('matching_result', {})
        metrics = data.get('metrics', {})
        
        analysis = {
            # Basic extraction metrics
            'metabolites_extracted': extraction_result.get('unique_metabolites', 0),
            'total_metabolites_found': extraction_result.get('total_metabolites', 0),
            'successful_chunks': extraction_result.get('successful_chunks', 0),
            'failed_chunks': extraction_result.get('failed_chunks', 0),
            
            # Database matching metrics
            'csv_database_size': matching_result.get('csv_database_size', 0),
            'matches_found': matching_result.get('matches_found', 0),
            'detection_rate': matching_result.get('detection_rate', 0),
            
            # Performance metrics (main focus)
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'accuracy': metrics.get('accuracy', 0),
            
            # Detailed metrics
            'true_positives': metrics.get('true_positives', 0),
            'false_positives': metrics.get('false_positives', 0),
            'false_negatives': metrics.get('false_negatives', 0),
            
            # Processing time breakdown
            'pdf_extraction_time': data.get('pdf_result', {}).get('extraction_time', 0),
            'llm_processing_time': extraction_result.get('processing_time', 0),
            'database_matching_time': matching_result.get('processing_time', 0),
            
            # LLM statistics
            'llm_statistics': extraction_result.get('llm_statistics', {}),
            
            # Matched biomarkers for detailed analysis
            'matched_biomarkers': matching_result.get('matched_biomarkers', [])
        }
        
        logger.info(f"Analysis complete for {model}: F1={analysis['f1_score']:.4f}, Recall={analysis['recall']:.4f}")
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze results for {model}: {e}")
        raise

def generate_comprehensive_report(results: Dict[str, Any], logger):
    """Generate comprehensive evaluation report"""
    logger.info("\n📊 COMPREHENSIVE MODEL EVALUATION REPORT")
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 80)
    
    successful_models = {k: v for k, v in results.items() if v.get('success', False)}
    failed_models = {k: v for k, v in results.items() if not v.get('success', False)}
    
    print(f"✅ Successful models: {len(successful_models)}/{len(results)}")
    print(f"❌ Failed models: {len(failed_models)}/{len(results)}")
    
    if successful_models:
        print(f"\n🏆 PERFORMANCE RANKING (by F1 Score)")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<35} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Time':<8}")
        print("-" * 80)
        
        # Sort by F1 score (descending)
        ranked_models = sorted(
            successful_models.items(),
            key=lambda x: x[1].get('f1_score', 0),
            reverse=True
        )
        
        for i, (model, data) in enumerate(ranked_models, 1):
            model_short = model.replace('meta-llama/', '').replace('moonshotai/', '')[:30]
            f1 = f"{data.get('f1_score', 0):.4f}"
            precision = f"{data.get('precision', 0):.4f}"
            recall = f"{data.get('recall', 0):.4f}"
            time_str = f"{data.get('total_time', 0):.1f}s"
            
            print(f"{i:<4} {model_short:<35} {f1:<10} {precision:<10} {recall:<10} {time_str:<8}")
        
        print(f"\n⚡ SPEED RANKING")
        print("-" * 50)
        print(f"{'Rank':<4} {'Model':<35} {'Time':<10}")
        print("-" * 50)
        
        # Sort by processing time (ascending)
        speed_ranked = sorted(
            successful_models.items(),
            key=lambda x: x[1].get('total_time', float('inf'))
        )
        
        for i, (model, data) in enumerate(speed_ranked, 1):
            model_short = model.replace('meta-llama/', '').replace('moonshotai/', '')[:30]
            time_str = f"{data.get('total_time', 0):.1f}s"
            print(f"{i:<4} {model_short:<35} {time_str:<10}")
        
        print(f"\n📊 DETAILED METRICS")
        print("-" * 100)
        print(f"{'Model':<35} {'Metabolites':<12} {'Matches':<8} {'Detection%':<12} {'Chunks':<8} {'LLM Time':<10}")
        print("-" * 100)
        
        for model, data in ranked_models:
            model_short = model.replace('meta-llama/', '').replace('moonshotai/', '')[:30]
            metabolites = data.get('metabolites_extracted', 0)
            matches = data.get('matches_found', 0)
            detection = f"{data.get('detection_rate', 0):.1%}"
            chunks = f"{data.get('successful_chunks', 0)}/{data.get('successful_chunks', 0) + data.get('failed_chunks', 0)}"
            llm_time = f"{data.get('llm_processing_time', 0):.1f}s"
            
            print(f"{model_short:<35} {metabolites:<12} {matches:<8} {detection:<12} {chunks:<8} {llm_time:<10}")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS")
        print("-" * 50)
        for model, data in failed_models.items():
            print(f"   • {model}: {data.get('error', 'Unknown error')}")
    
    # Save detailed results to JSON
    save_evaluation_results(results, logger)
    
    # Generate summary statistics
    if successful_models:
        generate_summary_statistics(successful_models, logger)

def save_evaluation_results(results: Dict[str, Any], logger):
    """Save detailed evaluation results to JSON"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"comprehensive_model_evaluation_{timestamp}.json"
    
    # Prepare results for JSON serialization
    evaluation_data = {
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_document': 'Wine-consumptionbiomarkers-HMDB.pdf',
        'ground_truth_database': 'urinary_wine_biomarkers.csv',
        'total_models_evaluated': len(results),
        'successful_models': len([r for r in results.values() if r.get('success', False)]),
        'failed_models': len([r for r in results.values() if not r.get('success', False)]),
        'model_results': results
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        logger.info(f"💾 Detailed evaluation results saved to: {filename}")
        print(f"\n💾 Detailed results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save evaluation results: {e}")

def generate_summary_statistics(successful_models: Dict[str, Any], logger):
    """Generate summary statistics for successful models"""
    print(f"\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    
    f1_scores = [data.get('f1_score', 0) for data in successful_models.values()]
    processing_times = [data.get('total_time', 0) for data in successful_models.values()]
    metabolites_counts = [data.get('metabolites_extracted', 0) for data in successful_models.values()]
    
    print(f"F1 Score Statistics:")
    print(f"   Average: {sum(f1_scores)/len(f1_scores):.4f}")
    print(f"   Best: {max(f1_scores):.4f}")
    print(f"   Worst: {min(f1_scores):.4f}")
    
    print(f"\nProcessing Time Statistics:")
    print(f"   Average: {sum(processing_times)/len(processing_times):.1f}s")
    print(f"   Fastest: {min(processing_times):.1f}s")
    print(f"   Slowest: {max(processing_times):.1f}s")
    
    print(f"\nMetabolite Extraction Statistics:")
    print(f"   Average: {sum(metabolites_counts)/len(metabolites_counts):.1f}")
    print(f"   Most: {max(metabolites_counts)}")
    print(f"   Least: {min(metabolites_counts)}")

def main():
    """Main evaluation function"""
    print("🧪 COMPREHENSIVE GROQ MODEL EVALUATION FOR FOODB PIPELINE")
    print("=" * 70)
    
    try:
        results = comprehensive_model_evaluation()
        
        print(f"\n🎉 EVALUATION COMPLETED!")
        print(f"📊 Check comprehensive_model_evaluation_*.json for detailed results")
        print(f"📋 Check comprehensive_model_evaluation.log for complete logs")
        print(f"📁 Check ./comprehensive_evaluation/ for individual model outputs")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
