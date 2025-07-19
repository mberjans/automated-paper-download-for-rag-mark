# 🔍 F1 Score Paradox Explained: Why Higher Recall Led to Lower F1

## ❓ **Your Question: F1 Score Contradiction**

You correctly observed this apparent contradiction:
- **Test 1 (137 chunks)**: 83.1% recall → F1 = 0.397
- **Test 2 (69 chunks)**: 74.6% recall → F1 = 0.406

**Higher recall but lower F1 score - why?**

---

## 📊 **Complete Data Analysis**

| Test | Chunks | Recall | Precision | F1 Score | Total Extracted | Biomarkers Found |
|------|--------|--------|-----------|----------|------------------|------------------|
| **1** | 137 | **83.1%** | **26.1%** | **0.397** | ~188 compounds | 49/59 |
| **2** | 69 | **74.6%** | **27.8%** | **0.406** | ~158 compounds | 44/59 |
| **3** | 86 | **74.6%** | **28.8%** | **0.415** | ~152 compounds | 44/59 |

---

## 💡 **The Key Insight: Precision-Recall Trade-off**

### **What Happened:**

#### **Test 1 (137 chunks, 500 chars each):**
- ✅ **Higher Recall**: Found 49/59 biomarkers (83.1%)
- ❌ **Lower Precision**: 49 correct out of ~188 total extracted (26.1%)
- **Result**: Extracted many false positives along with true biomarkers

#### **Test 2 (69 chunks, 1000 chars each):**
- ❌ **Lower Recall**: Found 44/59 biomarkers (74.6%)
- ✅ **Higher Precision**: 44 correct out of ~158 total extracted (27.8%)
- **Result**: Extracted fewer false positives, cleaner results

---

## 🧮 **F1 Score Mathematics**

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

### **Test 1 Calculation:**
```
Precision = 0.261, Recall = 0.831
F1 = 2 × (0.261 × 0.831) / (0.261 + 0.831)
F1 = 2 × 0.217 / 1.092 = 0.397
```

### **Test 2 Calculation:**
```
Precision = 0.278, Recall = 0.746
F1 = 2 × (0.278 × 0.746) / (0.278 + 0.746)
F1 = 2 × 0.207 / 1.024 = 0.405
```

**The precision drop (-1.7%) outweighed the recall gain (+8.5%) in the harmonic mean!**

---

## ⚖️ **Why This Happens: Chunk Size Effects**

### **Smaller Chunks (500 chars) → More Noise**
- **More granular extraction** → Captures isolated compound names
- **Less context** → Higher chance of false positives
- **More chunks** → More opportunities for noise extraction
- **Result**: Higher recall but lower precision

### **Larger Chunks (1000 chars) → Better Context**
- **More context per chunk** → Better compound identification
- **Fewer chunks** → Fewer opportunities for noise
- **Better filtering** → Cleaner extraction
- **Result**: Lower recall but higher precision

---

## 📈 **F1 Score Sensitivity to Precision**

F1 score is the **harmonic mean**, which is heavily penalized by low values:

| Scenario | Recall | Precision | F1 Score | Notes |
|----------|--------|-----------|----------|-------|
| **Actual Test 1** | 83.1% | 26.1% | **0.397** | High recall, low precision |
| **Actual Test 2** | 74.6% | 27.8% | **0.406** | Balanced |
| **Hypothetical** | 83.1% | 27.8% | **0.417** | High recall, same precision |

**If Test 1 had the same precision as Test 2, F1 would be 0.417 (highest)!**

---

## 🔍 **Root Cause Analysis**

### **Why Smaller Chunks Extract More False Positives:**

1. **Fragmented Context**: 500-char chunks break sentences mid-way
2. **Isolated Terms**: Compound-like words without proper context
3. **Higher Volume**: 137 chunks vs 69 chunks = 98% more extraction opportunities
4. **Less Filtering**: Shorter context makes it harder to distinguish real compounds

### **Example Scenario:**
```
Original text: "The wine contained resveratrol, which is beneficial for health..."

500-char chunk: "...contained resveratrol, which is beneficial..."
→ Might extract: "resveratrol", "beneficial" (false positive)

1000-char chunk: "The wine contained resveratrol, which is beneficial for health and has been studied extensively..."
→ Extracts: "resveratrol" (correct, with context)
```

---

## 📊 **Information Retrieval Principle**

This is a **classic precision-recall trade-off** in information retrieval:

### **High Recall Systems:**
- Cast a wide net
- Capture most relevant items
- But also capture many irrelevant items
- **Good for**: Comprehensive searches where missing items is costly

### **High Precision Systems:**
- Use strict criteria
- Return mostly relevant items
- But miss some relevant items
- **Good for**: Quality-focused applications where false positives are costly

### **F1 Score Balances Both:**
- Harmonic mean penalizes extreme imbalances
- Favors systems with balanced precision and recall
- **Best for**: Applications needing both quality and completeness

---

## 🎯 **Practical Implications**

### **For FOODB Pipeline:**

#### **Use Smaller Chunks (500 chars) When:**
- ✅ Completeness is critical (don't miss any biomarkers)
- ✅ You can post-process to filter false positives
- ✅ Recall is more important than precision

#### **Use Larger Chunks (1000 chars) When:**
- ✅ Quality is critical (minimize false positives)
- ✅ Processing time/cost is a concern
- ✅ Precision is more important than recall

#### **Optimal Strategy:**
- **Medium chunks (800 chars)**: Best F1 score (0.415)
- **Balanced approach**: Good recall (74.6%) with better precision (28.8%)

---

## ✅ **Conclusion**

**The F1 score decrease despite higher recall is completely normal and expected:**

1. **📈 Higher Recall (83.1% vs 74.6%)**: Smaller chunks found more biomarkers
2. **📉 Lower Precision (26.1% vs 27.8%)**: But also extracted more false positives
3. **⚖️ F1 Trade-off**: Harmonic mean penalized the precision drop more than it rewarded the recall gain
4. **🎯 Classic Pattern**: This is a well-known precision-recall trade-off in information retrieval

**This demonstrates the system is working correctly - it's capturing the fundamental trade-off between completeness (recall) and accuracy (precision) based on chunk size configuration.**

**The "paradox" is actually evidence of a well-functioning system that responds predictably to different parameter settings!** 🔍📊✅
