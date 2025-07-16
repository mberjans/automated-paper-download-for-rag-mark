import subprocess
import re
import os

def parse_query(query):
    patterns = {
        'Food Categories': r'(Fruits|Vegetables|Grains|Dairy|Meats|Seafood|Legumes|Nuts|Seeds|Herbs|Spices|Beverages)',
        'Organoleptic Properties': r'(Taste|Texture|Aroma|Color|Mouthfeel)',
        'Food Sources': r'(Plant-based|Animal-based|Synthetic|Fermented|Fortified foods)',
        'Functional Foods': r'(Probiotics|Prebiotics|Fortified foods|Nutraceuticals)',
    }
    results = {}
    for category, pattern in patterns.items():
        match = re.findall(pattern, query)
        if match:
            results[category] = match
    return results

def execute_script(file, directory, query):
    if query:
        parsed_query = parse_query(query)
        formatted_query = ' '.join([f"{k}: {', '.join(v)}" for k, v in parsed_query.items()])  # Format the parsed query
        print(f"Formatted query: {formatted_query}")  # For debugging, remove or replace with actual processing
    if file and directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        command = f'./Download.sh {file} {directory} "{formatted_query}"'
        print(f"Executing command: {command}")  # Add this line
        subprocess.run(command, shell=True)

# Example usage
file = "/path/to/your/file"
directory = "/path/to/your/directory"
query = "Your query here"

execute_script(file, directory, query)
