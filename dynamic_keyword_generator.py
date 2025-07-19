#!/usr/bin/env python3
"""
Dynamic Keyword Generation for Relevance Evaluation
Multiple approaches to generate keywords automatically instead of hardcoding
"""

import csv
import json
import requests
from typing import Dict, List, Set
from collections import Counter
import re

class DynamicKeywordGenerator:
    """Generate keywords dynamically from various sources"""
    
    def __init__(self):
        self.keyword_sources = {}
    
    def generate_from_csv_database(self, csv_file: str) -> Dict[str, float]:
        """Generate keywords from CSV database of known compounds"""
        print(f"📊 Generating keywords from CSV database: {csv_file}")
        
        keywords = {}
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                compounds = []
                
                for row in reader:
                    # Assuming 'Compound Name' column
                    compound_name = row.get('Compound Name', '').strip()
                    if compound_name:
                        compounds.append(compound_name.lower())
                
                print(f"✅ Loaded {len(compounds)} compounds from CSV")
                
                # Extract keywords from compound names
                for compound in compounds:
                    # Full compound names (high weight)
                    keywords[compound] = 1.0
                    
                    # Extract chemical suffixes and prefixes
                    words = compound.replace('-', ' ').split()
                    for word in words:
                        if len(word) > 3:  # Skip very short words
                            keywords[word] = 0.8
                    
                    # Extract chemical patterns
                    if 'glucoside' in compound:
                        keywords['glucoside'] = 1.0
                    if 'glucuronide' in compound:
                        keywords['glucuronide'] = 1.0
                    if 'sulfate' in compound:
                        keywords['sulfate'] = 1.0
                
                print(f"✅ Generated {len(keywords)} keywords from compounds")
                return keywords
                
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return {}
    
    def generate_from_pubchem_api(self, compound_list: List[str]) -> Dict[str, float]:
        """Generate keywords using PubChem API for compound synonyms"""
        print(f"🔬 Generating keywords from PubChem API...")
        
        keywords = {}
        
        for compound in compound_list[:5]:  # Limit for demo
            try:
                # Search PubChem for compound
                search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound}/synonyms/JSON"
                response = requests.get(search_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    synonyms = data.get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])
                    
                    # Add synonyms as keywords
                    for synonym in synonyms[:10]:  # Limit synonyms
                        if len(synonym) > 3 and synonym.isalpha():
                            keywords[synonym.lower()] = 0.9
                    
                    print(f"  ✅ {compound}: Found {len(synonyms)} synonyms")
                else:
                    print(f"  ❌ {compound}: API error {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ {compound}: Error {e}")
        
        print(f"✅ Generated {len(keywords)} keywords from PubChem")
        return keywords
    
    def generate_from_text_analysis(self, text_corpus: str) -> Dict[str, float]:
        """Generate keywords by analyzing existing text corpus"""
        print(f"📝 Generating keywords from text analysis...")
        
        # Chemical term patterns
        chemical_patterns = [
            r'\b\w*(?:yl|ol|ic|ine|ate|ide|ose)\b',  # Chemical suffixes
            r'\b\w*(?:phenol|flavon|anthocyan)\w*\b',  # Chemical families
            r'\b\w+(?:glucoside|glucuronide|sulfate)\b',  # Conjugates
        ]
        
        keywords = {}
        
        # Extract potential chemical terms
        for pattern in chemical_patterns:
            matches = re.findall(pattern, text_corpus, re.IGNORECASE)
            for match in matches:
                if len(match) > 4:  # Skip very short matches
                    keywords[match.lower()] = 0.7
        
        # Extract frequent scientific terms
        scientific_terms = [
            'metabolite', 'biomarker', 'compound', 'concentration',
            'detection', 'analysis', 'chromatography', 'spectrometry',
            'urinary', 'plasma', 'serum', 'excretion'
        ]
        
        for term in scientific_terms:
            if term.lower() in text_corpus.lower():
                keywords[term] = 0.8
        
        print(f"✅ Generated {len(keywords)} keywords from text analysis")
        return keywords
    
    def generate_from_ontology(self, ontology_source: str = "chebi") -> Dict[str, float]:
        """Generate keywords from chemical ontologies (simulated)"""
        print(f"🧬 Generating keywords from {ontology_source} ontology...")
        
        # Simulated ontology-based keywords (in real implementation, would query ChEBI, etc.)
        ontology_keywords = {
            # Chemical classes
            'phenolic_compound': 1.0,
            'flavonoid': 1.0,
            'anthocyanin': 1.0,
            'stilbene': 1.0,
            'tannin': 1.0,
            
            # Metabolic processes
            'glucuronidation': 0.9,
            'sulfation': 0.9,
            'methylation': 0.9,
            'hydroxylation': 0.9,
            
            # Analytical methods
            'liquid_chromatography': 0.8,
            'mass_spectrometry': 0.8,
            'tandem_ms': 0.8,
            
            # Biological matrices
            'urine': 0.7,
            'plasma': 0.7,
            'serum': 0.7,
            'blood': 0.7
        }
        
        print(f"✅ Generated {len(ontology_keywords)} keywords from ontology")
        return ontology_keywords
    
    def generate_from_machine_learning(self, training_texts: List[str], labels: List[bool]) -> Dict[str, float]:
        """Generate keywords using ML feature importance (simulated)"""
        print(f"🤖 Generating keywords using ML feature importance...")
        
        # Simulated ML-based keyword extraction
        # In real implementation, would use TF-IDF, word embeddings, etc.
        
        all_words = []
        relevant_words = []
        
        for text, is_relevant in zip(training_texts, labels):
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
            if is_relevant:
                relevant_words.extend(words)
        
        # Calculate word importance based on frequency in relevant vs all texts
        all_word_counts = Counter(all_words)
        relevant_word_counts = Counter(relevant_words)
        
        keywords = {}
        for word, relevant_count in relevant_word_counts.items():
            if len(word) > 3:  # Skip short words
                total_count = all_word_counts[word]
                importance = relevant_count / total_count if total_count > 0 else 0
                if importance > 0.1:  # Threshold for inclusion
                    keywords[word] = min(importance * 2, 1.0)  # Scale and cap
        
        print(f"✅ Generated {len(keywords)} keywords using ML")
        return keywords
    
    def combine_keyword_sources(self, *keyword_dicts: Dict[str, float]) -> Dict[str, float]:
        """Combine multiple keyword sources with weighted averaging"""
        print(f"🔄 Combining {len(keyword_dicts)} keyword sources...")
        
        combined = {}
        word_counts = Counter()
        
        # Collect all words and their scores
        for keyword_dict in keyword_dicts:
            for word, score in keyword_dict.items():
                if word not in combined:
                    combined[word] = 0.0
                combined[word] += score
                word_counts[word] += 1
        
        # Average the scores
        for word in combined:
            combined[word] /= word_counts[word]
        
        # Sort by importance
        sorted_keywords = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
        
        print(f"✅ Combined into {len(sorted_keywords)} unique keywords")
        return sorted_keywords
    
    def save_keywords(self, keywords: Dict[str, float], filename: str):
        """Save generated keywords to file"""
        with open(filename, 'w') as f:
            json.dump(keywords, f, indent=2)
        print(f"💾 Saved {len(keywords)} keywords to {filename}")
    
    def load_keywords(self, filename: str) -> Dict[str, float]:
        """Load keywords from file"""
        try:
            with open(filename, 'r') as f:
                keywords = json.load(f)
            print(f"📂 Loaded {len(keywords)} keywords from {filename}")
            return keywords
        except Exception as e:
            print(f"❌ Error loading keywords: {e}")
            return {}

def demonstrate_dynamic_keywords():
    """Demonstrate dynamic keyword generation"""
    print("🚀 Dynamic Keyword Generation Demonstration")
    print("=" * 50)
    
    generator = DynamicKeywordGenerator()
    
    # Method 1: From CSV database
    if True:  # Simulated - would use actual CSV
        print("\n1. 📊 CSV Database Method:")
        csv_keywords = {
            'resveratrol': 1.0, 'quercetin': 1.0, 'malvidin': 1.0,
            'glucoside': 1.0, 'glucuronide': 1.0, 'sulfate': 1.0,
            'anthocyanin': 1.0, 'catechin': 1.0, 'epicatechin': 1.0
        }
        print(f"   Generated {len(csv_keywords)} keywords from CSV")
    
    # Method 2: Text analysis
    print("\n2. 📝 Text Analysis Method:")
    sample_text = """
    The urinary metabolites of wine consumption included resveratrol glucuronide,
    quercetin sulfate, and various anthocyanin derivatives. These biomarkers were
    detected using LC-MS analysis and showed significant concentration changes.
    """
    text_keywords = generator.generate_from_text_analysis(sample_text)
    
    # Method 3: Ontology-based
    print("\n3. 🧬 Ontology Method:")
    ontology_keywords = generator.generate_from_ontology("chebi")
    
    # Method 4: ML-based (simulated)
    print("\n4. 🤖 Machine Learning Method:")
    training_texts = [
        "Wine contains resveratrol and quercetin compounds",
        "The study measured urinary metabolites after consumption",
        "References: Smith et al. 2020, Jones et al. 2019"
    ]
    labels = [True, True, False]  # First two are relevant
    ml_keywords = generator.generate_from_machine_learning(training_texts, labels)
    
    # Combine all sources
    print("\n5. 🔄 Combining All Sources:")
    combined_keywords = generator.combine_keyword_sources(
        csv_keywords, text_keywords, ontology_keywords, ml_keywords
    )
    
    # Show top keywords
    print(f"\n📋 Top 15 Combined Keywords:")
    for i, (word, score) in enumerate(list(combined_keywords.items())[:15]):
        print(f"   {i+1:2d}. {word:<20} {score:.3f}")
    
    # Save for future use
    generator.save_keywords(combined_keywords, "dynamic_keywords.json")
    
    return combined_keywords

if __name__ == "__main__":
    demonstrate_dynamic_keywords()
