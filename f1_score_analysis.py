#!/usr/bin/env python3
"""
Analyze the F1 score discrepancy: Why F1 decreased when recall increased
"""

def analyze_f1_discrepancy():
    """Analyze why F1 score decreased when recall increased"""
    print("🔍 F1 Score Discrepancy Analysis")
    print("=" * 50)
    
    # Data from the test runs (extracted from terminal outputs)
    test_data = [
        {
            'name': 'Test 1 (137 chunks, 500 chars)',
            'chunks': 137,
            'chunk_size': 500,
            'biomarkers_found': 49,
            'total_biomarkers': 59,
            'recall': 49/59,  # 83.1%
            'precision': 0.261,  # From terminal output
            'f1_score': 0.397,  # From terminal output
            'total_extracted': None  # Need to calculate
        },
        {
            'name': 'Test 2 (69 chunks, 1000 chars)',
            'chunks': 69,
            'chunk_size': 1000,
            'biomarkers_found': 44,
            'total_biomarkers': 59,
            'recall': 44/59,  # 74.6%
            'precision': 0.278,  # From terminal output
            'f1_score': 0.406,  # From terminal output
            'total_extracted': None  # Need to calculate
        },
        {
            'name': 'Test 3 (86 chunks, 800 chars)',
            'chunks': 86,
            'chunk_size': 800,
            'biomarkers_found': 44,
            'total_biomarkers': 59,
            'recall': 44/59,  # 74.6%
            'precision': 0.288,  # From terminal output
            'f1_score': 0.415,  # From terminal output
            'total_extracted': None  # Need to calculate
        }
    ]
    
    # Calculate total extracted compounds for each test
    for test in test_data:
        if test['precision'] > 0:
            test['total_extracted'] = int(test['biomarkers_found'] / test['precision'])
    
    print("📊 Test Results Comparison:")
    print(f"{'Test':<25} {'Recall':<8} {'Precision':<10} {'F1':<8} {'Extracted':<10}")
    print("-" * 70)
    
    for test in test_data:
        print(f"{test['name']:<25} {test['recall']:<8.3f} {test['precision']:<10.3f} {test['f1_score']:<8.3f} {test['total_extracted']:<10}")
    
    print(f"\n🔍 Key Observation:")
    print(f"   Test 1: Higher recall (0.831) but LOWER precision (0.261) → Lower F1 (0.397)")
    print(f"   Test 2: Lower recall (0.746) but HIGHER precision (0.278) → Higher F1 (0.406)")
    
    # Verify F1 calculations
    print(f"\n🧮 F1 Score Verification:")
    for test in test_data:
        calculated_f1 = 2 * (test['precision'] * test['recall']) / (test['precision'] + test['recall'])
        print(f"   {test['name'][:15]}: Reported={test['f1_score']:.3f}, Calculated={calculated_f1:.3f}")
    
    # Explain the phenomenon
    print(f"\n💡 Explanation:")
    print(f"   F1 = 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   ")
    print(f"   Test 1 (137 chunks): Found 49/59 biomarkers BUT extracted ~188 total compounds")
    print(f"   → High recall (83.1%) but low precision (26.1%)")
    print(f"   → Many false positives dilute the precision")
    print(f"   ")
    print(f"   Test 2 (69 chunks): Found 44/59 biomarkers from ~158 total compounds")
    print(f"   → Lower recall (74.6%) but higher precision (27.8%)")
    print(f"   → Fewer false positives, better precision")
    
    # Show the trade-off
    print(f"\n⚖️ Precision-Recall Trade-off:")
    print(f"   Smaller chunks (500 chars):")
    print(f"   ✅ Extract more biomarkers (higher recall)")
    print(f"   ❌ Extract more noise/false positives (lower precision)")
    print(f"   ❌ F1 score suffers due to precision drop")
    print(f"   ")
    print(f"   Larger chunks (1000 chars):")
    print(f"   ❌ Extract fewer biomarkers (lower recall)")
    print(f"   ✅ Extract less noise (higher precision)")
    print(f"   ✅ F1 score benefits from better precision")
    
    return test_data

def demonstrate_f1_sensitivity():
    """Demonstrate how F1 is sensitive to precision drops"""
    print(f"\n📈 F1 Score Sensitivity Analysis:")
    print("=" * 40)
    
    # Simulate the effect
    scenarios = [
        {'recall': 0.831, 'precision': 0.261, 'name': 'Test 1 (High recall, low precision)'},
        {'recall': 0.746, 'precision': 0.278, 'name': 'Test 2 (Balanced)'},
        {'recall': 0.831, 'precision': 0.278, 'name': 'Hypothetical (High recall, same precision)'}
    ]
    
    print(f"{'Scenario':<35} {'Recall':<8} {'Precision':<10} {'F1':<8}")
    print("-" * 65)
    
    for scenario in scenarios:
        f1 = 2 * (scenario['precision'] * scenario['recall']) / (scenario['precision'] + scenario['recall'])
        print(f"{scenario['name']:<35} {scenario['recall']:<8.3f} {scenario['precision']:<10.3f} {f1:<8.3f}")
    
    print(f"\n🎯 Key Insight:")
    print(f"   F1 score is the harmonic mean - it's heavily penalized by low precision")
    print(f"   Even with higher recall, if precision drops significantly, F1 will decrease")
    print(f"   This is exactly what happened: 83.1% recall vs 74.6% recall (+8.5%)")
    print(f"   But 26.1% precision vs 27.8% precision (-1.7%)")
    print(f"   The precision drop outweighed the recall gain in the harmonic mean")

if __name__ == "__main__":
    print("🔍 F1 Score Analysis: Why Higher Recall Led to Lower F1")
    print("=" * 60)
    
    # Analyze the discrepancy
    test_data = analyze_f1_discrepancy()
    
    # Demonstrate F1 sensitivity
    demonstrate_f1_sensitivity()
    
    print(f"\n✅ Conclusion:")
    print(f"   The F1 score decrease despite higher recall is due to:")
    print(f"   1. Smaller chunks extract more compounds (including false positives)")
    print(f"   2. Higher recall (83.1% vs 74.6%) but lower precision (26.1% vs 27.8%)")
    print(f"   3. F1 is harmonic mean - heavily penalized by precision drops")
    print(f"   4. The precision penalty outweighed the recall benefit")
    print(f"   ")
    print(f"   This is a classic precision-recall trade-off in information retrieval!")
