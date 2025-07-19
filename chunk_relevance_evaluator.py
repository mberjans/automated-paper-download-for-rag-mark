#!/usr/bin/env python3
"""
Chunk Relevance Evaluator for FOODB Pipeline
Evaluates the relevance of text chunks to metabolite extraction task
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class RelevanceScore:
    """Relevance score with breakdown"""
    total_score: float
    keyword_score: float
    section_score: float
    chemical_score: float
    context_score: float
    is_relevant: bool

class ChunkRelevanceEvaluator:
    """Evaluates chunk relevance to metabolite extraction task"""
    
    def __init__(self, relevance_threshold: float = 0.3):
        self.relevance_threshold = relevance_threshold
        
        # Metabolite-related keywords (weighted by importance)
        self.metabolite_keywords = {
            # High importance (weight: 1.0)
            'metabolite': 1.0, 'biomarker': 1.0, 'compound': 1.0,
            'phenolic': 1.0, 'flavonoid': 1.0, 'anthocyanin': 1.0,
            'catechin': 1.0, 'resveratrol': 1.0, 'quercetin': 1.0,
            'glucoside': 1.0, 'glucuronide': 1.0, 'sulfate': 1.0,
            
            # Medium importance (weight: 0.7)
            'urinary': 0.7, 'plasma': 0.7, 'serum': 0.7, 'urine': 0.7,
            'excretion': 0.7, 'concentration': 0.7, 'detection': 0.7,
            'analysis': 0.7, 'chromatography': 0.7, 'spectrometry': 0.7,
            
            # Lower importance (weight: 0.5)
            'wine': 0.5, 'grape': 0.5, 'polyphenol': 0.5, 'antioxidant': 0.5,
            'consumption': 0.5, 'intake': 0.5, 'dietary': 0.5,
            'chemical': 0.5, 'organic': 0.5, 'molecule': 0.5
        }
        
        # Chemical name patterns
        self.chemical_patterns = [
            r'\b\w+(?:yl|ol|ic|ine|ate|ide)\b',  # Chemical suffixes
            r'\b\d+[,-]\w+\b',  # Numbered compounds
            r'\b[A-Z][a-z]+-\d+\b',  # Compound-number patterns
            r'\b\w+(?:glucoside|glucuronide|sulfate)\b',  # Conjugated forms
        ]
        
        # Section relevance (higher = more relevant)
        self.section_weights = {
            'results': 1.0,
            'discussion': 0.9,
            'methods': 0.8,
            'materials': 0.8,
            'analysis': 0.9,
            'conclusion': 0.7,
            'introduction': 0.5,
            'abstract': 0.6,
            'references': 0.1,
            'acknowledgment': 0.1,
            'funding': 0.1,
            'author': 0.1
        }
        
        # Irrelevant content patterns
        self.irrelevant_patterns = [
            r'^\s*\d+\.\s*$',  # Just numbers
            r'^\s*[A-Z\s]+\s*$',  # All caps headers
            r'^\s*Figure\s+\d+',  # Figure captions
            r'^\s*Table\s+\d+',  # Table captions
            r'^\s*References?\s*$',  # Reference sections
            r'^\s*Acknowledgments?\s*$',  # Acknowledgment sections
        ]
    
    def evaluate_chunk_relevance(self, chunk: str, chunk_index: int = 0) -> RelevanceScore:
        """Evaluate the relevance of a chunk to metabolite extraction"""
        
        # Calculate individual scores
        keyword_score = self._calculate_keyword_score(chunk)
        section_score = self._calculate_section_score(chunk)
        chemical_score = self._calculate_chemical_score(chunk)
        context_score = self._calculate_context_score(chunk)
        
        # Check for irrelevant content
        if self._is_irrelevant_content(chunk):
            return RelevanceScore(
                total_score=0.0,
                keyword_score=0.0,
                section_score=0.0,
                chemical_score=0.0,
                context_score=0.0,
                is_relevant=False
            )
        
        # Weighted combination
        total_score = (
            keyword_score * 0.4 +      # 40% weight on keywords
            chemical_score * 0.3 +     # 30% weight on chemical patterns
            section_score * 0.2 +      # 20% weight on section context
            context_score * 0.1        # 10% weight on general context
        )
        
        is_relevant = total_score >= self.relevance_threshold
        
        return RelevanceScore(
            total_score=total_score,
            keyword_score=keyword_score,
            section_score=section_score,
            chemical_score=chemical_score,
            context_score=context_score,
            is_relevant=is_relevant
        )
    
    def _calculate_keyword_score(self, chunk: str) -> float:
        """Calculate score based on metabolite-related keywords"""
        chunk_lower = chunk.lower()
        total_score = 0.0
        word_count = len(chunk.split())
        
        for keyword, weight in self.metabolite_keywords.items():
            count = chunk_lower.count(keyword)
            if count > 0:
                # Score based on frequency and weight, normalized by chunk length
                total_score += (count * weight) / max(word_count / 100, 1)
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def _calculate_chemical_score(self, chunk: str) -> float:
        """Calculate score based on chemical name patterns"""
        total_matches = 0
        
        for pattern in self.chemical_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            total_matches += len(matches)
        
        # Normalize by chunk length
        word_count = len(chunk.split())
        score = min(total_matches / max(word_count / 50, 1), 1.0)
        
        return score
    
    def _calculate_section_score(self, chunk: str) -> float:
        """Calculate score based on section context"""
        chunk_lower = chunk.lower()
        
        # Look for section indicators
        max_weight = 0.5  # Default neutral weight
        
        for section, weight in self.section_weights.items():
            if section in chunk_lower:
                max_weight = max(max_weight, weight)
        
        return max_weight
    
    def _calculate_context_score(self, chunk: str) -> float:
        """Calculate score based on general scientific context"""
        scientific_indicators = [
            'study', 'research', 'analysis', 'experiment', 'data',
            'significant', 'correlation', 'concentration', 'measured',
            'detected', 'identified', 'quantified', 'observed'
        ]
        
        chunk_lower = chunk.lower()
        matches = sum(1 for indicator in scientific_indicators if indicator in chunk_lower)
        
        return min(matches / 10.0, 1.0)  # Normalize to 0-1
    
    def _is_irrelevant_content(self, chunk: str) -> bool:
        """Check if chunk contains irrelevant content"""
        for pattern in self.irrelevant_patterns:
            if re.match(pattern, chunk.strip(), re.IGNORECASE):
                return True
        
        # Check if chunk is too short or mostly numbers/punctuation
        if len(chunk.strip()) < 50:
            return True
        
        # Check if chunk is mostly non-alphabetic
        alpha_chars = sum(1 for c in chunk if c.isalpha())
        if alpha_chars / len(chunk) < 0.5:
            return True
        
        return False
    
    def filter_relevant_chunks(self, chunks: List[str]) -> Tuple[List[str], List[RelevanceScore]]:
        """Filter chunks to only include relevant ones"""
        relevant_chunks = []
        all_scores = []
        
        for i, chunk in enumerate(chunks):
            score = self.evaluate_chunk_relevance(chunk, i)
            all_scores.append(score)
            
            if score.is_relevant:
                relevant_chunks.append(chunk)
        
        return relevant_chunks, all_scores
    
    def get_relevance_report(self, chunks: List[str]) -> Dict:
        """Generate a detailed relevance report"""
        relevant_chunks, scores = self.filter_relevant_chunks(chunks)
        
        total_chunks = len(chunks)
        relevant_count = len(relevant_chunks)
        filtered_count = total_chunks - relevant_count
        
        avg_score = sum(score.total_score for score in scores) / len(scores) if scores else 0
        
        return {
            'total_chunks': total_chunks,
            'relevant_chunks': relevant_count,
            'filtered_chunks': filtered_count,
            'relevance_rate': relevant_count / total_chunks if total_chunks > 0 else 0,
            'average_relevance_score': avg_score,
            'threshold_used': self.relevance_threshold,
            'detailed_scores': [
                {
                    'chunk_index': i,
                    'total_score': score.total_score,
                    'keyword_score': score.keyword_score,
                    'chemical_score': score.chemical_score,
                    'section_score': score.section_score,
                    'context_score': score.context_score,
                    'is_relevant': score.is_relevant
                }
                for i, score in enumerate(scores)
            ]
        }

def demonstrate_relevance_evaluation():
    """Demonstrate the relevance evaluation system"""
    print("🔍 Chunk Relevance Evaluation Demonstration")
    print("=" * 50)
    
    # Sample chunks with different relevance levels
    test_chunks = [
        # High relevance
        "The urinary metabolites of wine consumption included resveratrol glucuronide, quercetin sulfate, and various anthocyanin derivatives. These biomarkers were detected using LC-MS analysis.",
        
        # Medium relevance  
        "Wine consumption has been associated with various health benefits. The polyphenolic compounds in red wine may contribute to cardiovascular protection.",
        
        # Low relevance
        "The study was conducted at the University of California. Participants were recruited through local advertisements and provided informed consent.",
        
        # Irrelevant
        "References: 1. Smith et al. (2020) 2. Jones et al. (2019) 3. Brown et al. (2018)"
    ]
    
    evaluator = ChunkRelevanceEvaluator(relevance_threshold=0.3)
    
    print("📊 Evaluating sample chunks:")
    print("-" * 40)
    
    for i, chunk in enumerate(test_chunks):
        score = evaluator.evaluate_chunk_relevance(chunk, i)
        
        print(f"\nChunk {i+1}: {chunk[:60]}...")
        print(f"  Total Score: {score.total_score:.3f}")
        print(f"  Keyword: {score.keyword_score:.3f} | Chemical: {score.chemical_score:.3f}")
        print(f"  Section: {score.section_score:.3f} | Context: {score.context_score:.3f}")
        print(f"  Relevant: {'✅ YES' if score.is_relevant else '❌ NO'}")
    
    # Generate report
    report = evaluator.get_relevance_report(test_chunks)
    
    print(f"\n📋 Relevance Report:")
    print(f"  Total chunks: {report['total_chunks']}")
    print(f"  Relevant chunks: {report['relevant_chunks']}")
    print(f"  Filtered out: {report['filtered_chunks']}")
    print(f"  Relevance rate: {report['relevance_rate']:.1%}")
    print(f"  Average score: {report['average_relevance_score']:.3f}")

if __name__ == "__main__":
    demonstrate_relevance_evaluation()
