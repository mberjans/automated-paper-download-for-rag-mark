# Written by Mahi and Omolola
# Code grabs pdfs from a file, extracts text, cleans up text, breaks them into sentences, then conatnate sentenes into stacks of 5. 
# number of sentences and number of stacks for each are printed

import os
from pdfminer.high_level import extract_text
import spacy
import language_tool_python
import math
import json
import csv
import re
import glob
import pandas as pd

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
# Load the LanguageTool grammar checker
tool = language_tool_python.LanguageTool('en-US')


# 1. PDF Processing and Sentence Correction Functions

def remove_references(text):
    """Removes the last reference section and citations from the text."""
    pattern = re.compile(
        r"""
        (
            ([Rr]eference[s]?|[Rr]EFERENCE[S]?|[Bb]ibliography|[Bb]IBLIOGRAPHY|[Ww]orks [Cc]ited|[Ww]ORKS [Cc]ITED|[Ee]ndnotes|[Ee]NDNOTES)\s*\n.*
        )
        """,
        re.VERBOSE | re.DOTALL
    )
    text = pattern.sub("", text)    
    return text

def remove_numbered_citations(text):
    """Removes numbered citations from the text."""
    citation_pattern = re.compile(
        r"""
        \[\d+(?:[,–]\s*\d+)*\]
        """,
        re.VERBOSE
    )
    text = citation_pattern.sub("", text)
    return text

def process_text_to_sentences(path_to_pdf):
    """Extracts text from a PDF, removes references and citations, and returns corrected sentences."""
    text = extract_text(path_to_pdf)
    text = remove_references(text)
    text = remove_numbered_citations(text)

    pattern = r'(\w+)-(\n)?(\w+)'
    # Replace hyphenated words with a single word (concatenating word parts)
    text = re.sub(pattern, r'\1\3', text)
    # Split the text into lines
    lines = text.split('\n')

    # Use a list comprehension to filter out empty lines and lines with one character
    filtered_lines = [line for line in lines if len(line.strip()) > 1]

    # Rejoin the filtered lines to form the cleaned text
    cleaned_text = '\n'.join(filtered_lines)
    cleaned_text = cleaned_text.replace("\n", " ")

    # Process the text with spaCy
    doc = nlp(cleaned_text)

    # Extract sentences
    sentences = [sent.text for sent in doc.sents]

    # Initialize a list to store corrected sentences
    corrected_sentences = []
    tool.disable_spellchecking()

    # Correct the grammar in each sentence
    for sentence in sentences:
        corrected = tool.correct(sentence)
        corrected_sentences.append(corrected)

    return corrected_sentences

def extract_title_from_path(file_path):
    """Extracts the file name without extension."""
    try:
        _, filename = os.path.split(file_path)
        title, _ = os.path.splitext(filename)
        return title
    except Exception as e:
        print(f"Error extracting title from '{file_path}': {e}")
        return None


# 2. CSV Line Concatenation Function

def concatenate_lines(input_csv, output_csv, window_size=5):
    """Concatenates lines from a CSV into consecutive groups of window_size."""
    
    df = pd.read_csv(input_csv, header=None)  # Read CSV without headers

    concatenated_rows = []
    for i in range(0, len(df), window_size):
        end_index = min(i + window_size, len(df))
        concatenated_text = "\n".join(df.iloc[i:end_index, 0])
        concatenated_rows.append([concatenated_text])

    result_df = pd.DataFrame(concatenated_rows)
    result_df.to_csv(output_csv, index=False, header=False)

# 3. CSV to JSONL Function

def csv_to_jsonl(input_csv, output_jsonl):
    """Converts a CSV file (with stacked sentences) to JSONL format."""
    
    # Open the CSV and read its rows
    with open(input_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        
        # Open the JSONL file for writing
        with open(output_jsonl, "w", encoding="utf-8") as jsonlfile:
            for row in reader:
                # Each row contains the concatenated text (5 sentences)
                concatenated_text = row[0]  # There's only one column with concatenated text
                
                # Create a JSON object with the key 'input'
                json_obj = {"input": concatenated_text}
                
                # Write the JSON object as a line in the JSONL file
                jsonlfile.write(json.dumps(json_obj) + "\n")
    
    print(f"Converted {input_csv} to {output_jsonl}")


# 4. Main Processing Function

def process_pdfs_and_concatenate(pdf_folder_path, output_folder_csv, output_folder_stacked, output_folder_jsonl, window_size=5):
    """Processes PDFs, extracts and corrects sentences, saves to CSV, and concatenates sentences."""
    
    os.makedirs(output_folder_csv, exist_ok=True)
    os.makedirs(output_folder_stacked, exist_ok=True)
    os.makedirs(output_folder_jsonl, exist_ok=True)

    pdf_files = glob.glob(pdf_folder_path + "/*.pdf")
    
    for pdf_file in pdf_files:
        # Extract and clean sentences from the PDF
        list_processed_sentences = process_text_to_sentences(pdf_file)
        pdf_name = extract_title_from_path(pdf_file)

        # Step 1: Count number of sentences (x)
        num_sentences = len(list_processed_sentences)
        print(f"{pdf_name} has {num_sentences} sentences.")

        # Step 2: Save the corrected sentences into a CSV
        output_file_csv = os.path.join(output_folder_csv, pdf_name + ".csv")
        with open(output_file_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, escapechar='\\')
            for s in list_processed_sentences:
                writer.writerow([s])

        # Step 3: Concatenate sentences in groups of window_size and save to another CSV
        output_file_stacked = os.path.join(output_folder_stacked, "stacked_" + pdf_name + ".csv")
        concatenate_lines(output_file_csv, output_file_stacked, window_size)

        # Step 4: Calculate number of stacks (y)
        num_stacks = math.ceil(num_sentences / window_size)
        print(f"{pdf_name} was stacked into {num_stacks} groups of {window_size} sentences each.")

        # Step 5: Convert the final stacked CSV to JSONL format
        output_file_jsonl = os.path.join(output_folder_jsonl, "stacked_" + pdf_name + ".jsonl")
        csv_to_jsonl(output_file_stacked, output_file_jsonl)

        # Output the stats
        print(f"Processed {pdf_name}: {num_sentences} sentences into {num_stacks} stacks.")

        print(f"Processed, concatenated, and converted: {pdf_file}")


# 4. Run the Full Workflow

pdf_folder = "....your file path to the pdf's......"
output_folder_csv = "..processed_csv......"
output_folder_stacked = "stack_5_sentences"
output_folder_jsonl = "stacked_5_sentences_jsonl"  # This is the folder for JSONL files

# Process PDFs and concatenate sentences into groups of 5, then save as JSONL
process_pdfs_and_concatenate(pdf_folder, output_folder_csv, output_folder_stacked, output_folder_jsonl, window_size=5)


