#This script automates the retrieval and processing of PubMed articles related to specific diseases (based on the choices you make with your keywords), 
# saving the results as CSV files 
# and then processing those CSVs into text files (the DOIs picked from the CSV is saved as text in a folder to be fed to download.sh). 
# Before use, 
# pip install pandas
# pip install biopython
# if you need few articles (1-2000) your email is not needed (eventual number is based on your keywords)
# if email is needed you'll get a prompt.


import os
import pandas as pd
from Bio import Entrez
import xml.etree.ElementTree as ET
import time

# CSV Retrieval Functions
def search_pubmed(query, max_results, year_range=None, article_type=None):
    Entrez.email = "your_email@example.com"  # Set your email address for identification
    if year_range:
        query += f" AND {year_range[0]}:{year_range[1]}[PDAT]"
    if article_type:
        query += f" AND {article_type}[Filter]"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"], int(record["Count"])

def fetch_article_details_csv(article_ids):
    Entrez.email = "your_email@example.com"  # Set your email address for identification
    handle = Entrez.efetch(db="pubmed", id=article_ids, rettype="csv", retmode="text")
    return handle.read()

def fetch_article_details_xml(article_ids):
    Entrez.email = "your_email@example.com"  # Set your email address for identification
    handle = Entrez.efetch(db="pubmed", id=article_ids, rettype="xml", retmode="text")
    data = handle.read()
    handle.close()
    return data

def count_pmc_full_text(xml_data):
    root = ET.fromstring(xml_data)
    pmc_count = 0
    for article in root.findall(".//PubmedArticle"):
        if article.find(".//PubmedData/ArticleIdList/ArticleId[@IdType='pmc']") is not None:
            pmc_count += 1
    return pmc_count

def save_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        f.write(data)

def retrieve_csvs():
    # Create the folder for storing CSV files
    folder_name = "MarkerDBinput2"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define a list of dictionaries for each disease
    disease_keywords = [
        {"name": "ADHD", "keywords": ["ADHD", "biomarkers", "compounds"]},
        {"name": "Alzheimer Disease", "keywords": ["Alzheimer Disease", "biomarkers", "compounds"]},
        {"name": "Dementia", "keywords": ["Dementia", "biomarkers", "plasma"]},
        {"name": "ADHD", "keywords": ["ADHD", "symptoms", "types"]},
        {"name": "Alzheimer Disease", "keywords": ["Alzheimer Disease", "types", "symptoms"]},
        {"name": "ADHD", "keywords": ["ADHD", "biomarkers", "urine"]},
        {"name": "Alzheimer Disease", "keywords": ["Alzheimer Disease", "biomarkers", "urine"]},
        {"name": "ADHD", "keywords": ["ADHD", "biomarkers", "plasma"]},
        {"name": "ADHD", "keywords": ["ADHD", "biomarkers", "signs"]},
        # Add more diseases with associated keywords as needed
    ]

    max_results = 10000  # Maximum number of results to retrieve
    year_range = (2000, 2024)  # Filter by Publication date for the articles (optional)
    article_type = None  # Filter by desired article type(s) e.g. Meta-Analysis, Systematic Review, Review, Clinical Trial (CT), Randomized CT
    retry_attempts = 5  # Define the number of retry attempts
    retry_delay = 10  # Delay between retry attempts (in seconds)

    # Attempt to retrieve PubMed results for each disease with retry logic
    for disease in disease_keywords:
        try:
            # Construct the query for the current disease
            query = " AND ".join(disease["keywords"])

            # Search PubMed for articles matching the query within the specified time frame and article type
            article_ids, total_results = search_pubmed(query, max_results, year_range, article_type)

            print(f"Total results for {disease['name']}: {total_results}")

            # Fetch details of the articles in CSV format
            csv_data = fetch_article_details_csv(article_ids)

            print(f"CSV data for {disease['name']}: {csv_data[:100]}...")

            # Save the retrieved data to a CSV file with the filename based on the disease name
            save_csv(csv_data, os.path.join(folder_name, f"{disease['name']}_results.csv"))

            # Fetch XML details of the articles to count PMC full text
            xml_data = fetch_article_details_xml(article_ids)
            pmc_count = count_pmc_full_text(xml_data)
            non_pmc_count = total_results - pmc_count

            print(f"Full text in PMC for {disease['name']}: {pmc_count}")
            print(f"Not in PMC for {disease['name']}: {non_pmc_count}")

            # Print success message with the actual number of results obtained
            print(f"{total_results} results for {disease['name']} successfully downloaded.")
        except Exception as e:
            # Print error message
            print(f"Error: {e}")
            if "429" in str(e):  # Check if error is 429 (rate limit exceeded)
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(retry_delay)
                continue  # Retry the request
            else:
                break  # Exit the loop if the error is not 429 or if the maximum number of retry attempts is reached

    # Remove duplicates from the list of all article IDs
    # all_article_ids = list(set(all_article_ids))

# CSV Processing Function
def process_csv_files():
    # Directory containing the CSV files
    input_dir = 'MarkerDBinput2'

    # Directory to save the TXT files
    output_dir = 'MarkerDBtxt'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each CSV file in the directory
    for csv_file in os.listdir(input_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(input_dir, csv_file)
            
            print(f"Processing file: {csv_file}")
            
            # Read the CSV file without headers
            try:
                df = pd.read_csv(csv_path, header=None)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
            
            # Check if Column K (11th column, index 10) exists
            if df.shape[1] < 11:
                print(f"Column K not found in {csv_file}.")
                continue
            
            # Get the content of Column K
            column_k_data = df.iloc[:, 10].dropna().reset_index(drop=True)
            
            # Skip if no data in Column K
            if column_k_data.empty:
                print(f"No content in Column K of {csv_file}.")
                continue
            
            print(f"Found {len(column_k_data)} rows in Column K of {csv_file}.")
            
            # Split data if more than 1000 rows
            num_files = (len(column_k_data) // 1000) + 1
            base_filename = os.path.splitext(csv_file)[0]
            
            for i in range(num_files):
                start_row = i * 1000
                end_row = start_row + 1000
                chunk = column_k_data[start_row:end_row]
                
                if not chunk.empty:
                    # Create a filename based on the original CSV file name and chunk index
                    txt_file = f"{base_filename}_{i+1}.txt"
                    txt_path = os.path.join(output_dir, txt_file)
                    
                    # Write the data to a text file
                    with open(txt_path, 'w') as f:
                        for doi in chunk:
                            f.write(f"{doi}\n")
                    
                    print(f"Written {len(chunk)} rows to {txt_file}")
                else:
                    print(f"No more data to write for {csv_file} in chunk {i+1}.")
            
            print(f"Finished processing {csv_file}.\n")

# Main Script
if __name__ == "__main__":
    retrieve_csvs()
    process_csv_files()
    os.system('./Download.sh')
