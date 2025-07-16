#!/usr/bin/env python3
"""
Comprehensive Performance Summary
Generate a complete summary of all FOODB pipeline performance and accuracy testing
"""

import json
import time
from pathlib import Path

def load_performance_reports():
    """Load all performance reports"""
    print("📊 Loading Performance Reports")
    print("=" * 30)
    
    reports = {}
    
    # Look for performance report files
    report_files = list(Path('.').glob('*Performance_Report_*.json'))
    
    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            if 'sentence_generation' in data:
                reports['sentence_generation'] = data
                print(f"✅ Loaded sentence generation report: {report_file}")
            elif 'performance_test' in data:
                reports['triple_extraction'] = data
                print(f"✅ Loaded triple extraction report: {report_file}")
                
        except Exception as e:
            print(f"⚠️ Error loading {report_file}: {e}")
    
    return reports

def generate_comprehensive_summary():
    """Generate comprehensive performance summary"""
    print("\n📋 FOODB Pipeline - Comprehensive Performance Summary")
    print("=" * 60)
    
    reports = load_performance_reports()
    
    if not reports:
        print("❌ No performance reports found")
        return
    
    # Initialize summary data
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_date': time.strftime('%Y-%m-%d'),
        'pipeline_components': {},
        'overall_metrics': {},
        'fallback_system_performance': {},
        'recommendations': []
    }
    
    print("\n🔍 ANALYSIS RESULTS")
    print("=" * 20)
    
    # Analyze sentence generation
    if 'sentence_generation' in reports:
        sg_data = reports['sentence_generation']
        
        print("\n📝 Sentence Generation Analysis:")
        
        if 'sentence_generation' in sg_data:
            sg_perf = sg_data['sentence_generation']
            print(f"  ✅ Success Rate: {sg_perf['success_rate']:.1%}")
            print(f"  ⚡ Avg Response Time: {sg_perf['avg_response_time']:.2f}s")
            print(f"  🚀 Throughput: {sg_perf['throughput']:.1f} req/s")
            
            summary['pipeline_components']['sentence_generation'] = {
                'success_rate': sg_perf['success_rate'],
                'avg_response_time': sg_perf['avg_response_time'],
                'throughput': sg_perf['throughput'],
                'status': 'excellent' if sg_perf['success_rate'] > 0.95 else 'good'
            }
        
        if 'accuracy_testing' in sg_data:
            acc_data = sg_data['accuracy_testing']
            print(f"  🎯 Overall Accuracy: {acc_data['overall_accuracy']:.1%}")
            print(f"  🏆 Perfect Extractions: {acc_data['perfect_cases']}/5")
            
            summary['pipeline_components']['sentence_generation']['accuracy'] = acc_data['overall_accuracy']
        
        if 'stress_testing' in sg_data:
            stress_data = sg_data['stress_testing']
            print(f"  🔥 Stress Success Rate: {stress_data['success_rate']:.1%}")
            print(f"  🔄 Provider Switches: {stress_data['provider_switches']}")
            
            summary['fallback_system_performance']['stress_test'] = {
                'success_rate': stress_data['success_rate'],
                'provider_switches': stress_data['provider_switches'],
                'handled_rate_limits': stress_data['final_stats']['rate_limited_requests'] > 0
            }
    
    # Analyze triple extraction
    if 'triple_extraction' in reports:
        te_data = reports['triple_extraction']
        
        print("\n🔗 Triple Extraction Analysis:")
        
        if 'performance_test' in te_data:
            te_perf = te_data['performance_test']
            print(f"  ✅ Success Rate: {te_perf['success_rate']:.1%}")
            print(f"  ⚡ Avg Response Time: {te_perf['avg_response_time']:.2f}s")
            print(f"  🚀 Throughput: {te_perf['throughput']:.1f} req/s")
            print(f"  🔄 Provider Switches: {te_perf['provider_switches']}")
            
            summary['pipeline_components']['triple_extraction'] = {
                'success_rate': te_perf['success_rate'],
                'avg_response_time': te_perf['avg_response_time'],
                'throughput': te_perf['throughput'],
                'provider_switches': te_perf['provider_switches'],
                'status': 'excellent' if te_perf['success_rate'] > 0.95 else 'good'
            }
        
        if 'accuracy_test' in te_data:
            te_acc = te_data['accuracy_test']
            print(f"  🎯 Average Accuracy: {te_acc['avg_accuracy']:.1%}")
            print(f"  🏆 Perfect Extractions: {te_acc['perfect_cases']}/{te_acc['total_cases']}")
            
            summary['pipeline_components']['triple_extraction']['accuracy'] = te_acc['avg_accuracy']
    
    # Calculate overall metrics
    print("\n📊 OVERALL PIPELINE METRICS")
    print("=" * 30)
    
    # Success rates
    success_rates = []
    response_times = []
    throughputs = []
    
    for component, data in summary['pipeline_components'].items():
        if 'success_rate' in data:
            success_rates.append(data['success_rate'])
        if 'avg_response_time' in data:
            response_times.append(data['avg_response_time'])
        if 'throughput' in data:
            throughputs.append(data['throughput'])
    
    if success_rates:
        avg_success_rate = sum(success_rates) / len(success_rates)
        print(f"  🎯 Average Success Rate: {avg_success_rate:.1%}")
        summary['overall_metrics']['avg_success_rate'] = avg_success_rate
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        print(f"  ⚡ Average Response Time: {avg_response_time:.2f}s")
        summary['overall_metrics']['avg_response_time'] = avg_response_time
    
    if throughputs:
        avg_throughput = sum(throughputs) / len(throughputs)
        print(f"  🚀 Average Throughput: {avg_throughput:.1f} req/s")
        summary['overall_metrics']['avg_throughput'] = avg_throughput
    
    # Fallback system analysis
    print("\n🛡️ FALLBACK SYSTEM ANALYSIS")
    print("=" * 30)
    
    fallback_working = False
    rate_limits_handled = False
    
    if 'stress_test' in summary['fallback_system_performance']:
        stress_data = summary['fallback_system_performance']['stress_test']
        fallback_working = stress_data['provider_switches'] > 0
        rate_limits_handled = stress_data['handled_rate_limits']
    
    # Check triple extraction switches too
    if 'triple_extraction' in summary['pipeline_components']:
        te_switches = summary['pipeline_components']['triple_extraction'].get('provider_switches', 0)
        if te_switches > 0:
            fallback_working = True
    
    print(f"  🔄 Fallback System Active: {'✅' if fallback_working else '❌'}")
    print(f"  🚨 Rate Limits Handled: {'✅' if rate_limits_handled else '❌'}")
    print(f"  🛡️ System Resilience: {'✅ Excellent' if fallback_working and rate_limits_handled else '⚠️ Needs Testing'}")
    
    summary['fallback_system_performance']['overall_status'] = {
        'active': fallback_working,
        'handles_rate_limits': rate_limits_handled,
        'resilience_rating': 'excellent' if fallback_working and rate_limits_handled else 'needs_testing'
    }
    
    # Generate recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 20)
    
    recommendations = []
    
    # Performance recommendations
    if avg_success_rate < 0.95:
        recommendations.append("Consider optimizing API error handling for higher success rates")
        print("  ⚠️ Consider optimizing API error handling for higher success rates")
    
    if avg_response_time > 0.5:
        recommendations.append("Response times could be improved with request optimization")
        print("  ⚠️ Response times could be improved with request optimization")
    
    # Accuracy recommendations
    sg_accuracy = summary['pipeline_components'].get('sentence_generation', {}).get('accuracy', 1.0)
    te_accuracy = summary['pipeline_components'].get('triple_extraction', {}).get('accuracy', 1.0)
    
    if sg_accuracy < 0.8:
        recommendations.append("Sentence generation accuracy needs improvement - consider prompt optimization")
        print("  ⚠️ Sentence generation accuracy needs improvement - consider prompt optimization")
    
    if te_accuracy < 0.8:
        recommendations.append("Triple extraction accuracy needs improvement - consider model fine-tuning")
        print("  ⚠️ Triple extraction accuracy needs improvement - consider model fine-tuning")
    
    # Positive recommendations
    if not recommendations:
        recommendations.append("System performing excellently - ready for production deployment")
        print("  ✅ System performing excellently - ready for production deployment")
    
    if fallback_working:
        recommendations.append("Fallback system working well - provides excellent resilience")
        print("  ✅ Fallback system working well - provides excellent resilience")
    
    summary['recommendations'] = recommendations
    
    # Overall assessment
    print("\n🎉 OVERALL ASSESSMENT")
    print("=" * 20)
    
    excellent_performance = avg_success_rate > 0.95 and avg_response_time < 0.5
    good_accuracy = sg_accuracy > 0.75 and te_accuracy > 0.8
    resilient_system = fallback_working and rate_limits_handled
    
    if excellent_performance and good_accuracy and resilient_system:
        overall_rating = "EXCELLENT"
        print("  🏆 EXCELLENT: System ready for production deployment")
    elif avg_success_rate > 0.9 and good_accuracy:
        overall_rating = "GOOD"
        print("  ✅ GOOD: System performing well with minor optimization opportunities")
    else:
        overall_rating = "NEEDS_IMPROVEMENT"
        print("  ⚠️ NEEDS IMPROVEMENT: Address performance or accuracy issues before production")
    
    summary['overall_assessment'] = {
        'rating': overall_rating,
        'excellent_performance': excellent_performance,
        'good_accuracy': good_accuracy,
        'resilient_system': resilient_system,
        'production_ready': excellent_performance and good_accuracy and resilient_system
    }
    
    # Save comprehensive summary
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    summary_file = f"FOODB_Pipeline_Comprehensive_Summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Comprehensive summary saved: {summary_file}")
    
    return summary

def display_final_results():
    """Display final test results summary"""
    print("\n🎯 FINAL PERFORMANCE AND ACCURACY RESULTS")
    print("=" * 50)
    
    summary = generate_comprehensive_summary()
    
    if summary:
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print(f"  Test Date: {summary['test_date']}")
        print(f"  Components Tested: {len(summary['pipeline_components'])}")
        print(f"  Overall Rating: {summary['overall_assessment']['rating']}")
        print(f"  Production Ready: {'✅' if summary['overall_assessment']['production_ready'] else '❌'}")
        
        print(f"\n🔑 KEY METRICS:")
        if 'avg_success_rate' in summary['overall_metrics']:
            print(f"  Success Rate: {summary['overall_metrics']['avg_success_rate']:.1%}")
        if 'avg_response_time' in summary['overall_metrics']:
            print(f"  Response Time: {summary['overall_metrics']['avg_response_time']:.2f}s")
        if 'avg_throughput' in summary['overall_metrics']:
            print(f"  Throughput: {summary['overall_metrics']['avg_throughput']:.1f} req/s")
        
        print(f"\n🛡️ FALLBACK SYSTEM:")
        fallback_status = summary['fallback_system_performance']['overall_status']
        print(f"  Active: {'✅' if fallback_status['active'] else '❌'}")
        print(f"  Handles Rate Limits: {'✅' if fallback_status['handles_rate_limits'] else '❌'}")
        print(f"  Resilience: {fallback_status['resilience_rating'].title()}")
        
        return summary
    
    return None

def main():
    """Main function to generate comprehensive summary"""
    print("📊 FOODB Pipeline - Comprehensive Performance Analysis")
    print("=" * 60)
    
    try:
        summary = display_final_results()
        
        if summary:
            print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
            
            if summary['overall_assessment']['production_ready']:
                print(f"\n✅ CONCLUSION: FOODB Pipeline is PRODUCTION READY!")
                print(f"  • High reliability and performance")
                print(f"  • Good accuracy for compound extraction")
                print(f"  • Robust fallback system active")
                print(f"  • Rate limiting resilience verified")
            else:
                print(f"\n⚠️ CONCLUSION: Additional optimization recommended")
                print(f"  • Review recommendations above")
                print(f"  • Address performance or accuracy issues")
                print(f"  • Re-test before production deployment")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
