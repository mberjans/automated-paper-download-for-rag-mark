#!/chunk/bin/env python
# pubmed_searcher.py - Script 2: Search PubMed/PMC and retrieve metadata

import pandas as pd
import os
import time
import json
import logging
from tqdm import tqdm
import requests
from urllib.parse import quote
import xml.etree.ElementTree as ET
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pubmed_search.log"),
        logging.StreamHandler()
    ]
)

class PubMedSearcher:
    def __init__(self, synonyms_csv, output_dir="pubmed_results", 
                 terms_to_add=[# Disease-specific mechanism terms
    "apoptosis induction", "cell cycle arrest", "COX-2 inhibition",
    "AMPK activation", "insulin sensitivity", "glucose metabolism",
    "angiogenesis inhibition", "LDL oxidation", "BDNF expression",
    "neuroinflammation", "dopaminergic signaling", "oxidative stress",
    # General mechanism terms
    "mechanism of action", "enzyme inhibition", "molecular target",
    "gene expression", "receptor binding", "pathway modulation",
    "anti-inflammatory mechanism", "kinase", "cytokine", "affinity",
    # Action verbs (useful but general)
    "inhibits", "binds", "downregulates", "upregulates", "modulates" "toxicity"]):
        self.synonyms_csv = synonyms_csv
        self.output_dir = output_dir
        self.compounds_df = None
        self.terms_to_add = terms_to_add
        self.email = "lolatemmy@gmail.com"  # Replace with your email for NCBI
        self.tool = "compound_bioactivity_retrieval"
        self.api_key = "313a24c282709267c07794adb44712fa7d08"  # Optional: Add your NCBI API key here for higher rate limits
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create a directory for search results
        self.results_dir = os.path.join(self.output_dir, "search_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
    def load_data(self):
        """Load the CSV file with compound synonyms"""
        logging.info(f"Loading compound data from {self.synonyms_csv}")
        self.compounds_df = pd.read_csv(self.synonyms_csv)
        logging.info(f"Loaded {len(self.compounds_df)} compounds with synonyms")
        return self.compounds_df
        
    def create_search_query(self, compound_names, max_synonyms=5, pub_types=None):
        """Create a PubMed search query using compound names and relevant terms"""
        # Limit to max_synonyms to avoid overly complex queries
        if len(compound_names) > max_synonyms:
            compound_names = compound_names[:max_synonyms]
            
        # Create compound part of the query (compound OR synonym OR synonym...)
        compound_query = " OR ".join([f'"{name}"' for name in compound_names])
        compound_query = f"({compound_query})"
        
        # Create bioactivity terms part (term1 OR term2 OR...)
        bioactivity_query = " OR ".join([f'"{term}"' for term in self.terms_to_add])
        bioactivity_query = f"({bioactivity_query})"
        
        # Define publication types to filter by
        # pub_types = ["Review", "Systematic Review", "Randomized Controlled Trial", "Books and Documents"]
    
        # Publication types filter (if specified)
        pub_type_query = ""
        if pub_types and len(pub_types) > 0:
            pub_type_query = " OR ".join([f'"{ptype}"[Publication Type]' for ptype in pub_types])
            pub_type_query = f" AND ({pub_type_query})"
        # Combine both parts
        full_query = f"{compound_query} AND {bioactivity_query}{pub_type_query}"
        
        return full_query
        
    def search_pubmed(self, query, retmax=100):
        """Search PubMed using E-utilities"""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
            "email": self.email,
            "tool": self.tool
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            results = response.json()
            id_list = results["esearchresult"].get("idlist", [])
            
            return id_list
            
        except Exception as e:
            logging.error(f"Error searching PubMed: {str(e)}")
            return []
            
    def fetch_paper_metadata(self, pmid_list):
        """Fetch metadata for PubMed IDs using E-utilities"""
        if not pmid_list:
            return []
            
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        
        # Convert list to comma-separated string
        pmids_str = ",".join(pmid_list)
        
        params = {
            "db": "pubmed",
            "id": pmids_str,
            "retmode": "json",
            "email": self.email,
            "tool": self.tool
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for pmid in pmid_list:
                if pmid in data["result"]:
                    paper_data = data["result"][pmid]
                    
                    # Check if paper has DOI
                    doi = None
                    for id_obj in paper_data.get("articleids", []):
                        if id_obj["idtype"] == "doi":
                            doi = id_obj["value"]
                            break
                            
                    # Check if available in PMC
                    pmc_id = None
                    for id_obj in paper_data.get("articleids", []):
                        if id_obj["idtype"] == "pmc":
                            pmc_id = id_obj["value"]
                            break
                            
                    results.append({
                        "pmid": pmid,
                        "doi": doi,
                        "pmc_id": pmc_id,
                        "title": paper_data.get("title", ""),
                        "pub_date": paper_data.get("pubdate", ""),
                        "journal": paper_data.get("fulljournalname", ""),
                        "authors": ", ".join([author.get("name", "") for author in paper_data.get("authors", [])]),
                        "has_abstract": True  # Assume it has an abstract, we'll check in the next step
                    })
                    
            return results
            
        except Exception as e:
            logging.error(f"Error fetching paper metadata: {str(e)}")
            return []
            
    def check_pmc_availability(self, pmc_id):
        """Check if a paper is available in PMC Open Access subset"""
        if not pmc_id:
            return False
            
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
            
            # If we can fetch it, it's available
            if response.status_code == 200:
                return True
                
            return False
            
        except Exception:
            return False
            
    def process_compounds(self, start_idx=0, batch_size=10, papers_per_compound=20, pub_types=None):
        """Process compounds to search for relevant papers"""
        if self.compounds_df is None:
            self.load_data()
            
        logging.info("Searching PubMed for compound bioactivity papers...")
        
        # Check for existing cache
        cache_file = os.path.join(self.output_dir, "search_cache.pkl")
        search_results = {}
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                search_results = pickle.load(f)
                
        # Process compounds in batches with progress bar
        total_compounds = len(self.compounds_df)
        
        for i in tqdm(range(start_idx, total_compounds, batch_size), 
                      desc="Searching PubMed"):
            batch = self.compounds_df.iloc[i:min(i+batch_size, total_compounds)]
            
            for _, row in batch.iterrows():
                compound_id = row['public_id']
                
                # Skip if already processed
                if compound_id in search_results:
                    continue
                    
                # Parse synonyms
                synonyms = row['synonyms'].split('|')
                
                # Create search query
                query = self.create_search_query(synonyms, pub_types=pub_types)
                
                # Search PubMed
                pmid_list = self.search_pubmed(query, retmax=papers_per_compound)
                
                if not pmid_list:
                    # No results found
                    search_results[compound_id] = {
                        "compound_id": compound_id,
                        "compound_name": row['original_name'],
                        "normalized_name": row['normalized_name'],
                        "query": query,
                        "papers": []
                    }
                    continue
                    
                # Fetch metadata for papers
                papers = self.fetch_paper_metadata(pmid_list)
                
                # Check PMC availability for papers with PMC IDs
                for paper in papers:
                    if paper["pmc_id"]:
                        paper["in_pmc_oa"] = self.check_pmc_availability(paper["pmc_id"])
                    else:
                        paper["in_pmc_oa"] = False
                        
                # Store results
                search_results[compound_id] = {
                    "compound_id": compound_id,
                    "compound_name": row['original_name'],
                    "normalized_name": row['normalized_name'],
                    "query": query,
                    "papers": papers
                }
                
                # Save individual result file
                result_file = os.path.join(self.results_dir, f"{compound_id}_results.json")
                with open(result_file, "w") as f:
                    json.dump(search_results[compound_id], f, indent=2)
                    
                # Be nice to NCBI
                time.sleep(0.34)  # Stay under 3 requests per second
                
            # Update cache after each batch
            with open(cache_file, "wb") as f:
                pickle.dump(search_results, f)
                
        # Generate summary report
        self.generate_summary(search_results)
        
        return search_results
    
    def filter_open_access_papers(self, input_file="papers_to_retrieve1.csv", output_file="papers_to_retrieve.csv"):
        """Filters out papers that are not in PubMed Central Open Access and saves to a new CSV."""
        # Load the papers DataFrame from the file
        papers_df = pd.read_csv(os.path.join(self.output_dir, input_file))

        # Filter out papers where 'in_pmc_oa' is False
        filtered_papers_df = papers_df[papers_df["in_pmc_oa"] == True]

        # Save the filtered DataFrame to a new CSV file
        filtered_papers_df.to_csv(os.path.join(self.output_dir, output_file), index=False)
        logging.info(f"Filtered papers saved to {os.path.join(self.output_dir, output_file)}")
       
    def generate_summary(self, search_results):
        """Generate a summary of search results"""
        summary_rows = []
        
        for compound_id, result in search_results.items():
            papers_count = len(result["papers"])
            pmc_papers = sum(1 for paper in result["papers"] if paper.get("in_pmc_oa", False))
            
            summary_rows.append({
                "compound_id": compound_id,
                "compound_name": result["compound_name"],
                "papers_found": papers_count,
                "pmc_available": pmc_papers
            })
            
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "search_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Create papers list for retrieval
        paper_rows = []
        
        for result in search_results.values():
            for paper in result["papers"]:
                if paper.get("doi"):  # Only include papers with DOI
                    paper_rows.append({
                        "pmid": paper["pmid"],
                        "doi": paper["doi"],
                        "pmc_id": paper["pmc_id"],
                        "in_pmc_oa": paper.get("in_pmc_oa", False),
                        "title": paper["title"],
                        "compound_id": result["compound_id"],
                        "compound_name": result["compound_name"]
                    })
                    
        # Create papers DataFrame
        papers_df = pd.DataFrame(paper_rows)
        
        # Remove duplicates (same paper might be found for multiple compounds)
        papers_df = papers_df.drop_duplicates(subset=["doi"])
        
        # Save papers list
        papers_file = os.path.join(self.output_dir, "papers_to_retrieve1.csv")
        papers_df.to_csv(papers_file, index=False)

        # Call the filtering method after saving the full papers list
        self.filter_open_access_papers()  # This will create "papers_to_retrieve.csv"

        # Save the summary and papers list
        logging.info(f"Summary saved to {summary_file}")
        logging.info(f"Papers list saved to {papers_file}")
        logging.info(f"Total unique papers to retrieve: {len(papers_df)}")

def main():
    input_file = "/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/normalized_compounds/compounds_with_synonyms.csv"  # Change to your input file
    
    # Define publication types to filter by
    pub_types = ["Review", "Systematic Review", "Randomized Controlled Trial"]
    
    # Check if the file exists
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found!")
        return
        
    searcher = PubMedSearcher(input_file)
    searcher.load_data()
    
    # Process compounds (start from index 0, process in batches of 10)
    searcher.process_compounds(start_idx=0, batch_size=10, papers_per_compound=20, pub_types=pub_types)
    
    logging.info("PubMed search completed!")

if __name__ == "__main__":
    main()