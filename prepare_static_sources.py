# prepare_static_sources.py

import pandas as pd
import os

# Create data directory if it does not exist to prevent FileNotFoundError
if not os.path.exists("data"):
    os.makedirs("data")

def process_csv_to_json(csv_path, json_path):
    """
    Reads a CSV file, standardizes the format, and exports it to JSON.
    """
    try:
        print(f"Processing: {csv_path}")
        
        # Load CSV using utf-8-sig to automatically handle Byte Order Mark (BOM)
        # sep=None allows pandas to automatically detect comma vs semicolon delimiters
        df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
        
        # Standardize column headers to lowercase snake_case for consistent mapping
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Specific handling for the CIQUAL database to convert French decimals
        if "ciqual" in csv_path.lower():
            for col in df.columns:
                # If column is text, attempt to replace commas with dots for math operations
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.replace(',', '.').astype(float)
                    except (ValueError, AttributeError):
                        # Continue if the column is strictly text (like food names)
                        continue 

        # Export the dataframe to JSON format using a list of records
        # The indent parameter ensures the output is human-readable
        df.to_json(json_path, orient="records", indent=2)
        print(f"Successfully saved to {json_path}\n")
        
    except Exception as e:
        print(f"Failed to process {csv_path}. Error: {e}")

if __name__ == "__main__":
    # Process the three core datasets: Reference values, food composition, and dietary sources
    process_csv_to_json("data/DRVs_Adults.csv", "data/DRVs_Adults.json")
    process_csv_to_json("data/Nutritional_Database_Ciqual.csv", "data/Nutritional_Database_Ciqual.json")
    process_csv_to_json("data/Dietary_Sources.csv", "data/Dietary_Sources.json")