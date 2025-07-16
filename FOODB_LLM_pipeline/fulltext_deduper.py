import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(
    filename='vectorized_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParagraphChecker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the paragraph checker with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.texts = []
        self.metadata = []
        
    def load_jsonl_file(self, file_path: str, text_field: str = 'input') -> List[Dict]:
        """
        Load data from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            text_field: Field name containing the text to process
            
        Returns:
            List of dictionaries from the JSONL file
        """
        logger.info(f"Loading JSONL file: {file_path}")
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        item = json.loads(line.strip())
                        if text_field in item:
                            data.append(item)
                        else:
                            logger.warning(f"Line {line_num}: Missing '{text_field}' field")
                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_num}: Invalid JSON - {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise
            
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data
    
    def vectorize_texts(self, data: List[Dict], text_field: str = 'input') -> np.ndarray:
        """
        Vectorize input texts using sentence transformers.
        
        Args:
            data: List of dictionaries containing text data
            text_field: Field name containing the input text to vectorize
            
        Returns:
            Numpy array of embeddings
        """
        # Extract texts and metadata
        self.texts = [item[text_field] for item in data]
        self.metadata = [item for item in data]
        
        logger.info(f"Vectorizing {len(self.texts)} texts...")
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            self.texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def find_duplicates(self, similarity_threshold: float = 0.85) -> List[Tuple[int, int, float]]:
        """
        Find duplicate pairs based on cosine similarity threshold.
        
        Args:
            similarity_threshold: Cosine similarity threshold for considering duplicates
            
        Returns:
            List of tuples (index1, index2, similarity_score)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run vectorize_texts first.")
        
        logger.info(f"Finding duplicates with threshold: {similarity_threshold}")
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # Find pairs above threshold (excluding diagonal)
        duplicates = []
        n = len(self.embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= similarity_threshold:
                    duplicates.append((i, j, similarity))
        
        # Sort by similarity score (descending)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates
    
    def remove_duplicates(self, similarity_threshold: float = 0.85) -> Tuple[List[Dict], List[Tuple[int, int, float]]]:
        """
        Remove duplicates while keeping track of what was removed.
        
        Args:
            similarity_threshold: Cosine similarity threshold for considering duplicates
            
        Returns:
            Tuple of (unique_items, removed_duplicates)
        """
        duplicates = self.find_duplicates(similarity_threshold)
        
        # Keep track of indices to remove
        indices_to_remove = set()
        
        # For each duplicate pair, mark the second one for removal
        for idx1, idx2, score in duplicates:
            if idx1 not in indices_to_remove and idx2 not in indices_to_remove:
                indices_to_remove.add(idx2)  # Keep the first occurrence
        
        # Create filtered dataset
        unique_items = []
        removed_items = []
        
        for i, item in enumerate(self.metadata):
            if i not in indices_to_remove:
                unique_items.append(item)
            else:
                removed_items.append((i, item))
        
        logger.info(f"Removed {len(removed_items)} duplicates, kept {len(unique_items)} unique items")
        
        return unique_items, duplicates
    
    def save_results(self, unique_items: List[Dict], output_file: str = 'deduped_fulltext.jsonl'):
        """
        Save the deduplicated results to a JSONL file.
        
        Args:
            unique_items: List of unique items to save
            output_file: Output file path
        """
        logger.info(f"Saving {len(unique_items)} unique items to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as file:
            for item in unique_items:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"Results saved to {output_file}")
    
    def process_file(self, 
                    input_file: str, 
                    output_file: str = 'deduped_fulltext.jsonl',
                    text_field: str = 'input',
                    similarity_threshold: float = 0.85) -> Dict:
        """
        Complete processing pipeline: load, vectorize, and deduplicate.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            text_field: Field containing text to process
            similarity_threshold: Similarity threshold for duplicates
            
        Returns:
            Dictionary with processing statistics
        """
        # Load data
        data = self.load_jsonl_file(input_file, text_field)
        original_count = len(data)
        
        # Vectorize
        self.vectorize_texts(data, text_field)
        
        # Remove duplicates
        unique_items, duplicates = self.remove_duplicates(similarity_threshold)
        
        # Save results
        self.save_results(unique_items, output_file)
        
        # Return statistics
        stats = {
            'original_count': original_count,
            'unique_count': len(unique_items),
            'duplicates_removed': original_count - len(unique_items),
            'duplicate_pairs_found': len(duplicates),
            'similarity_threshold': similarity_threshold
        }
        
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize the checker
    checker = ParagraphChecker()
    
    # Process the file - MODIFY THESE PATHS AS NEEDED
    try:
        stats = checker.process_file(
            input_file='/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/retrieved_papers/fulltext_output.jsonl',          # INPUT FILE PATH - change this to your file location
            output_file='deduped_fulltext.jsonl', # OUTPUT FILE PATH - change this to where you want results saved
            text_field='input',                   # Field containing text to analyze
            similarity_threshold=0.85
        )
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Original items: {stats['original_count']}")
        print(f"Unique items: {stats['unique_count']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")
        print(f"Duplicate pairs found: {stats['duplicate_pairs_found']}")
        print(f"Similarity threshold: {stats['similarity_threshold']}")
        print(f"Deduplication rate: {stats['duplicates_removed']/stats['original_count']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        print(f"Error: {e}")

# Advanced usage example
def analyze_duplicates(checker, duplicates, top_n=10):
    """
    Analyze the top duplicate pairs found.
    """
    print(f"\nTop {top_n} duplicate pairs:")
    print("-" * 60)
    
    for i, (idx1, idx2, score) in enumerate(duplicates[:top_n]):
        print(f"\n{i+1}. Similarity: {score:.3f}")
        print(f"Text 1 (index {idx1}): {checker.texts[idx1][:100]}...")
        print(f"Text 2 (index {idx2}): {checker.texts[idx2][:100]}...")
        print("-" * 60)