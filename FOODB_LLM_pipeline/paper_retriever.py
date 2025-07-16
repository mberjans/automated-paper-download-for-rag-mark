#!/chunk/bin/env python
# paper_retriever.py - Script 3: Retrieve full-text or abstracts and save as XML

import pandas as pd
import os
import time
import logging
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
from urllib.parse import quote
import re
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_retrieval.log"),
        logging.StreamHandler()
    ]
)

class PaperRetriever:
    def __init__(self, papers_csv, output_dir="retrieved_papers"):
        self.papers_csv = papers_csv
        self.output_dir = output_dir
        self.papers_df = None
        self.email = "otfatoku@ualberta.ca"  # Replace with your email for NCBI
        self.tool = "compound_bioactivity_retrieval"
        self.api_key = "02fdd0c7a55493ee69854f33ffbc1893c009"  # Optional: Add your NCBI API key here for higher rate limits
        
        # Create output directories
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.fulltext_dir = os.path.join(self.output_dir, "fulltext")
        if not os.path.exists(self.fulltext_dir):
            os.makedirs(self.fulltext_dir)
            
        self.abstract_dir = os.path.join(self.output_dir, "abstracts")
        if not os.path.exists(self.abstract_dir):
            os.makedirs(self.abstract_dir)
            
    def load_data(self):
        """Load the CSV file with papers to retrieve"""
        logging.info(f"Loading papers data from {self.papers_csv}")
        self.papers_df = pd.read_csv(self.papers_csv)
        logging.info(f"Loaded {len(self.papers_df)} papers to retrieve")
        return self.papers_df
        
    def sanitize_doi(self, doi):
        """Sanitize DOI for use in filenames"""
        if not doi:
            return ""
        return re.sub(r'[\\/*?:"<>|]', "_", doi)
        
    def fetch_pmc_fulltext(self, pmc_id):
        """Fetch full text from PubMed Central"""
        if not pmc_id:
            return None
            
        # Remove PMC prefix if present
        pmc_id = pmc_id.replace("PMC", "")
        
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml",
            "email": self.email,
            "tool": self.tool
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                return response.text
                
            return None
            
        except Exception as e:
            logging.error(f"Error fetching PMC fulltext for {pmc_id}: {str(e)}")
            return None
            
    def fetch_pubmed_abstract(self, pmid):
        """Fetch abstract from PubMed"""
        if not pmid:
            return None
            
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "email": self.email,
            "tool": self.tool
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                return response.text
                
            return None
            
        except Exception as e:
            logging.error(f"Error fetching abstract for PMID {pmid}: {str(e)}")
            return None
            
    def process_paper(self, paper_row):
        """Process a single paper - fetch fulltext or abstract"""
        pmid = paper_row['pmid']
        doi = paper_row['doi']
        pmc_id = paper_row['pmc_id']
        in_pmc_oa = paper_row['in_pmc_oa']
        
        sanitized_doi = self.sanitize_doi(doi)
        if not sanitized_doi:
            return False
            
        # Check if we've already processed this paper
        fulltext_file = os.path.join(self.fulltext_dir, f"{sanitized_doi}.xml")
        abstract_file = os.path.join(self.abstract_dir, f"{sanitized_doi}.xml")
        
        if os.path.exists(fulltext_file):
            logging.info(f"Fulltext already exists for DOI {doi}")
            return True
            
        if os.path.exists(abstract_file):
            logging.info(f"Abstract already exists for DOI {doi}")
            return True
            
        # Try to get fulltext first if available in PMC Open Access
        if pmc_id and in_pmc_oa:
            fulltext_xml = self.fetch_pmc_fulltext(pmc_id)
            
            if fulltext_xml:
                with open(fulltext_file, "w", encoding="utf-8") as f:
                    f.write(fulltext_xml)
                return True
                
        # Fall back to abstract if fulltext not available
        abstract_xml = self.fetch_pubmed_abstract(pmid)
        
        if abstract_xml:
            with open(abstract_file, "w", encoding="utf-8") as f:
                f.write(abstract_xml)
            return True
            
        logging.warning(f"Failed to retrieve either fulltext or abstract for DOI {doi}")
        return False
        
    def process_papers(self, max_workers=5):
        """Process all papers to retrieve fulltext or abstracts"""
        if self.papers_df is None:
            self.load_data()
            
        logging.info("Retrieving paper content...")
        
        success_count = 0
        failure_count = 0
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for _, row in self.papers_df.iterrows():
                futures.append(executor.submit(self.process_paper, row))
                
            # Process results as they complete
            for future in tqdm(futures, desc="Retrieving papers"):
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1
                    
        logging.info(f"Retrieval completed. Success: {success_count}, Failed: {failure_count}")
        
        # Create summary of retrieved papers
        self.generate_summary()
        
    def generate_summary(self):
        """Generate a summary of retrieved papers"""
        # Get lists of retrieved papers
        fulltext_files = os.listdir(self.fulltext_dir)
        abstract_files = os.listdir(self.abstract_dir)
        
        # Extract DOIs from filenames
        fulltext_dois = [file[:-4] for file in fulltext_files if file.endswith('.xml')]
        abstract_dois = [file[:-4] for file in abstract_files if file.endswith('.xml')]
        
        # Create summary dataframe
        summary_rows = []
        
        for doi in set(fulltext_dois + abstract_dois):
            has_fulltext = doi in fulltext_dois
            has_abstract = doi in abstract_dois
            
            # Find compound info from original dataframe
            compound_info = self.papers_df[self.papers_df['doi'].apply(
                lambda x: self.sanitize_doi(x) == doi
            )]
            
            compound_id = ""
            compound_name = ""
            title = ""
            
            if not compound_info.empty:
                compound_id = compound_info.iloc[0].get('compound_id', '')
                compound_name = compound_info.iloc[0].get('compound_name', '')
                title = compound_info.iloc[0].get('title', '')
            
            summary_rows.append({
                'doi': doi,
                'compound_id': compound_id,
                'compound_name': compound_name,
                'title': title,
                'has_fulltext': has_fulltext,
                'has_abstract': has_abstract
            })
        
        # Create DataFrame and save
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = os.path.join(self.output_dir, "retrieval_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            
            logging.info(f"Retrieved {len(fulltext_dois)} full-text papers and {len(abstract_dois)} abstracts")
            logging.info(f"Retrieval summary saved to {summary_file}")
        else:
            logging.warning("No papers were retrieved.")
def main():
    input_file = "/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/pubmed_results/papers_to_retrieve.csv"  # Change this to your input file path
    
    # Check if the file exists
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found!")
        return
    
    retriever = PaperRetriever(input_file)
    retriever.load_data()
    
    # Process papers with 5 parallel workers (adjust based on your system)
    retriever.process_papers(max_workers=5)
    
    logging.info("Paper retrieval completed!")

if __name__ == "__main__":
    main()
