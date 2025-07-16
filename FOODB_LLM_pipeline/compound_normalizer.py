#!/chunk/bin/env python
# compound_normalizer.py - Script 1: Normalize compounds and fetch synonyms
import pandas as pd
import re
import os
import time
import requests
from tqdm import tqdm
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("normalizer.log"),
        logging.StreamHandler()
    ]
)

class CompoundNormalizer:
    def __init__(self, input_csv, output_dir="normalized_compounds"):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.compounds_df = None
        self.normalized_df = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def load_data(self):
        """Load the CSV file with compound data"""
        logging.info(f"Loading compound data from {self.input_csv}")
        self.compounds_df = pd.read_csv(self.input_csv)
        logging.info(f"Loaded {len(self.compounds_df)} compounds")
        return self.compounds_df
        
    def normalize_name(self, name):
        """Normalize compound name by standardizing formatting"""
        if pd.isna(name):
            return ""
            
        # Convert to string if not already
        name = str(name)
        
        # Remove unnecessary spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Standardize brackets
        name = re.sub(r'\(([^()]*)\)', r'(\1)', name)  # Standardize parentheses
        
        # Standardize hyphens and dashes
        name = re.sub(r'[\u2010-\u2015]', '-', name)  # Convert various dash types to hyphen
        
        # Standardize quotes
        name = re.sub(r'[\u2018\u2019\u201C\u201D]', "'", name)
        
        # Remove starting/ending quotes or brackets if unmatched
        if name.count('(') != name.count(')'):
            name = name.replace('(', '').replace(')', '')
            
        # Standardize special characters specific to chemical nomenclature
        name = name.replace("''", "'").replace('""', '"')
        
        return name
        
    def process_compounds(self):
        """Process all compounds to normalize names"""
        if self.compounds_df is None:
            self.load_data()
            
        logging.info("Normalizing compound names...")
        
        # Copy the dataframe and add normalized name column
        self.normalized_df = self.compounds_df.copy()
        self.normalized_df['normalized_name'] = self.normalized_df['name'].apply(self.normalize_name)
        
        # Save intermediate results
        normalized_path = os.path.join(self.output_dir, "normalized_compounds.csv")
        self.normalized_df.to_csv(normalized_path, index=False)
        logging.info(f"Normalized compounds saved to {normalized_path}")
        
        return self.normalized_df
        
    def fetch_synonyms(self, compound_name, compound_id):
        """Fetch synonyms from PubChem"""
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
        
        try:
            # First try to get the PubChem CID
            search_url = f"{base_url}/{compound_name}/cids/JSON"
            response = requests.get(search_url)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            
            if 'IdentifierList' not in data or 'CID' not in data['IdentifierList']:
                return []
                
            cid = data['IdentifierList']['CID'][0]
            
            # Now get synonyms using the CID
            synonyms_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
            syn_response = requests.get(synonyms_url)
            
            if syn_response.status_code != 200:
                return []
                
            syn_data = syn_response.json()
            
            if 'InformationList' in syn_data and 'Information' in syn_data['InformationList']:
                synonyms = syn_data['InformationList']['Information'][0].get('Synonym', [])
                return synonyms
                
            return []
            
        except Exception as e:
            logging.warning(f"Error fetching synonyms for {compound_name} (ID: {compound_id}): {str(e)}")
            return []
            
    def batch_fetch_synonyms(self, batch_size=50, max_compounds=None):
        """Fetch synonyms for all compounds in batches"""
        if self.normalized_df is None:
            self.process_compounds()
            
        logging.info("Fetching synonyms for compounds...")
        
        # Initialize synonym dictionary
        synonyms_dict = {}
        
        # Determine how many compounds to process
        total_compounds = len(self.normalized_df) if max_compounds is None else min(max_compounds, len(self.normalized_df))
        compounds_to_process = self.normalized_df.iloc[:total_compounds]
        
        synonyms_cache_file = os.path.join(self.output_dir, "synonyms_cache.pkl")
        
        # Check if we have a cache file
        if os.path.exists(synonyms_cache_file):
            logging.info("Loading existing synonyms cache...")
            with open(synonyms_cache_file, 'rb') as f:
                synonyms_dict = pickle.load(f)
            
        # Process compounds in batches with progress bar
        for i in tqdm(range(0, len(compounds_to_process), batch_size), desc="Fetching synonyms"):
            batch = compounds_to_process.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                compound_id = row['public_id']
                compound_name = row['normalized_name']
                
                # Skip if we already have this compound in cache
                if compound_id in synonyms_dict:
                    continue
                    
                if pd.isna(compound_name) or compound_name == "":
                    synonyms_dict[compound_id] = []
                    continue
                    
                # Fetch synonyms
                synonyms = self.fetch_synonyms(compound_name, compound_id)
                synonyms_dict[compound_id] = synonyms
                
                # Be nice to the API
                time.sleep(0.2)
                
            # Save cache after each batch
            with open(synonyms_cache_file, 'wb') as f:
                pickle.dump(synonyms_dict, f)
                
        # Create a DataFrame with synonyms
        result_rows = []
        
        for _, row in compounds_to_process.iterrows():
            compound_id = row['public_id']
            compound_name = row['normalized_name']
            compound_class = row['klass']
            
            synonyms = synonyms_dict.get(compound_id, [])
            
            # Add the original and normalized names to synonyms if not already present
            all_names = set([row['name'], compound_name] + synonyms)
            all_names = [name for name in all_names if name and not pd.isna(name)]
            
            result_rows.append({
                'public_id': compound_id,
                'original_name': row['name'],
                'normalized_name': compound_name,
                'class': compound_class,
                'synonyms': '|'.join(all_names)
            })
            
        result_df = pd.DataFrame(result_rows)
        result_path = os.path.join(self.output_dir, "compounds_with_synonyms.csv")
        result_df.to_csv(result_path, index=False)
        logging.info(f"Compounds with synonyms saved to {result_path}")
        
        return result_df

def main():
    input_file = "/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/grab_bioactivities.csv"  # Change this to your input file path
    
    # Check if the file exists
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found!")
        return
        
    normalizer = CompoundNormalizer(input_file)
    normalizer.load_data()
    normalizer.process_compounds()
    
    # Fetch synonyms (limiting to 1000 compounds for testing)
    # Remove the max_compounds parameter to process all compounds
    normalizer.batch_fetch_synonyms(batch_size=50, max_compounds=1000)
    
    logging.info("Compound normalization and synonym expansion completed!")

if __name__ == "__main__":
    main()