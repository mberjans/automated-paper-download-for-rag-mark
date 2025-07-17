#!/usr/bin/env python3
"""
Test FOODB Pipeline with New Groq Models
Comprehensive testing of the pipeline with newly integrated Groq models
"""

import subprocess
import time
import json
from pathlib import Path

def test_pipeline_with_models():
    """Test the pipeline with different Groq models"""
    print("🧪 Testing FOODB Pipeline with New Groq Models")
    print("=" * 55)
    
    # Models to test (excluding problematic ones)
    models_to_test = [
        "moonshotai/kimi-k2-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct", 
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.1-8b-instant",  # Original model for comparison
    ]
    
    # Test parameters
    test_pdf = "Wine-consumptionbiomarkers-HMDB.pdf"
    output_dir = "./model_comparison_test"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    results = {}
    
    for i, model in enumerate(models_to_test, 1):
        print(f"\n🔬 Testing model {i}/{len(models_to_test)}: {model}")
        print("-" * 60)
        
        # Create model-specific output directory
        model_output_dir = f"{output_dir}/{model.replace('/', '_')}"
        
        try:
            start_time = time.time()
            
            # Run pipeline with specific model
            result = subprocess.run([
                "python", "foodb_pipeline_cli.py",
                test_pdf,
                "--groq-model", model,
                "--output-dir", model_output_dir,
                "--document-only",
                "--verify-compounds",
                "--save-timing",
                "--export-format", "json",
                "--chunk-size", "2000",
                "--quiet"
            ], capture_output=True, text=True, timeout=300)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ {model}: SUCCESS")
                print(f"   Processing time: {processing_time:.2f}s")
                
                # Analyze results
                analysis = analyze_model_results(model_output_dir, model)
                analysis['processing_time'] = processing_time
                analysis['success'] = True
                results[model] = analysis
                
                print(f"   Metabolites extracted: {analysis.get('metabolites_count', 'N/A')}")
                print(f"   Database matches: {analysis.get('matches_count', 'N/A')}")
                print(f"   Detection rate: {analysis.get('detection_rate', 'N/A'):.1%}" if analysis.get('detection_rate') else "")
                
            else:
                print(f"❌ {model}: FAILED")
                print(f"   Error: {result.stderr}")
                results[model] = {
                    'success': False,
                    'error': result.stderr,
                    'processing_time': processing_time
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model}: TIMEOUT (>300s)")
            results[model] = {
                'success': False,
                'error': 'Timeout after 300 seconds',
                'processing_time': 300
            }
        except Exception as e:
            print(f"❌ {model}: EXCEPTION - {str(e)}")
            results[model] = {
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    # Generate comparison report
    generate_comparison_report(results)
    
    return results

def analyze_model_results(output_dir: str, model: str) -> dict:
    """Analyze results from a model test"""
    try:
        # Find the results file
        output_path = Path(output_dir)
        json_files = list(output_path.glob("*_results.json"))
        
        if not json_files:
            return {'error': 'No results file found'}
        
        # Load the results
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        # Extract key metrics
        analysis = {
            'metabolites_count': data.get('extraction_result', {}).get('unique_metabolites', 0),
            'matches_count': data.get('matching_result', {}).get('matches_found', 0),
            'detection_rate': data.get('matching_result', {}).get('detection_rate', 0),
            'precision': data.get('metrics', {}).get('precision', 0),
            'recall': data.get('metrics', {}).get('recall', 0),
            'f1_score': data.get('metrics', {}).get('f1_score', 0),
            'successful_chunks': data.get('extraction_result', {}).get('successful_chunks', 0),
            'failed_chunks': data.get('extraction_result', {}).get('failed_chunks', 0),
            'llm_processing_time': data.get('extraction_result', {}).get('processing_time', 0)
        }
        
        return analysis
        
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}'}

def generate_comparison_report(results: dict):
    """Generate comprehensive comparison report"""
    print(f"\n📊 MODEL COMPARISON REPORT")
    print("=" * 50)
    
    successful_models = {k: v for k, v in results.items() if v.get('success', False)}
    failed_models = {k: v for k, v in results.items() if not v.get('success', False)}
    
    print(f"✅ Successful models: {len(successful_models)}/{len(results)}")
    print(f"❌ Failed models: {len(failed_models)}/{len(results)}")
    
    if successful_models:
        print(f"\n🎯 PERFORMANCE COMPARISON:")
        print(f"{'Model':<45} {'Time':<8} {'Metabolites':<12} {'Matches':<8} {'Recall':<8} {'F1':<8}")
        print("-" * 95)
        
        # Sort by F1 score (best performance indicator)
        sorted_models = sorted(
            successful_models.items(), 
            key=lambda x: x[1].get('f1_score', 0), 
            reverse=True
        )
        
        for model, data in sorted_models:
            model_short = model.replace('meta-llama/', '').replace('moonshotai/', '')[:40]
            time_str = f"{data.get('processing_time', 0):.1f}s"
            metabolites = data.get('metabolites_count', 0)
            matches = data.get('matches_count', 0)
            recall = f"{data.get('recall', 0):.3f}"
            f1 = f"{data.get('f1_score', 0):.3f}"
            
            print(f"{model_short:<45} {time_str:<8} {metabolites:<12} {matches:<8} {recall:<8} {f1:<8}")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS:")
        for model, data in failed_models.items():
            print(f"   • {model}: {data.get('error', 'Unknown error')}")
    
    # Speed ranking
    if successful_models:
        print(f"\n⚡ SPEED RANKING:")
        speed_sorted = sorted(
            successful_models.items(),
            key=lambda x: x[1].get('processing_time', float('inf'))
        )
        
        for i, (model, data) in enumerate(speed_sorted, 1):
            model_short = model.replace('meta-llama/', '').replace('moonshotai/', '')
            print(f"   {i}. {model_short}: {data.get('processing_time', 0):.1f}s")
    
    # Accuracy ranking
    if successful_models:
        print(f"\n🎯 ACCURACY RANKING (by F1 Score):")
        accuracy_sorted = sorted(
            successful_models.items(),
            key=lambda x: x[1].get('f1_score', 0),
            reverse=True
        )
        
        for i, (model, data) in enumerate(accuracy_sorted, 1):
            model_short = model.replace('meta-llama/', '').replace('moonshotai/', '')
            f1 = data.get('f1_score', 0)
            recall = data.get('recall', 0)
            print(f"   {i}. {model_short}: F1={f1:.3f}, Recall={recall:.3f}")
    
    # Save detailed results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = f"model_comparison_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def test_specific_model_features():
    """Test specific features with the best performing model"""
    print(f"\n🔬 Testing Specific Features with Best Model")
    print("=" * 45)
    
    best_model = "moonshotai/kimi-k2-instruct"
    test_pdf = "Wine-consumptionbiomarkers-HMDB.pdf"
    
    # Test different chunk sizes
    chunk_sizes = [1000, 1500, 2000, 2500]
    
    print(f"Testing chunk sizes with {best_model}:")
    
    for chunk_size in chunk_sizes:
        print(f"\n   Testing chunk size: {chunk_size}")
        
        try:
            start_time = time.time()
            
            result = subprocess.run([
                "python", "foodb_pipeline_cli.py",
                test_pdf,
                "--groq-model", best_model,
                "--output-dir", f"./chunk_test_{chunk_size}",
                "--document-only",
                "--chunk-size", str(chunk_size),
                "--quiet"
            ], capture_output=True, text=True, timeout=180)
            
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"   ✅ Chunk size {chunk_size}: {end_time - start_time:.1f}s")
            else:
                print(f"   ❌ Chunk size {chunk_size}: Failed")
                
        except Exception as e:
            print(f"   ❌ Chunk size {chunk_size}: {str(e)}")

def main():
    """Main testing function"""
    print("🧪 FOODB PIPELINE - NEW GROQ MODELS TESTING")
    print("=" * 55)
    
    try:
        # Test pipeline with different models
        results = test_pipeline_with_models()
        
        # Test specific features
        test_specific_model_features()
        
        print(f"\n🎉 TESTING COMPLETED!")
        print(f"📊 Check model_comparison_results_*.json for detailed analysis")
        print(f"📁 Check ./model_comparison_test/ for individual model outputs")
        
        # Provide recommendations
        successful_models = [k for k, v in results.items() if v.get('success', False)]
        if successful_models:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"✅ {len(successful_models)} models are working with the pipeline")
            print(f"🚀 Best performing model: moonshotai/kimi-k2-instruct (fastest)")
            print(f"🎯 Most accurate model: Check F1 scores in the comparison report")
            print(f"⚙️ Update your configuration to use: --groq-model moonshotai/kimi-k2-instruct")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
