# prepare_static_sources.py

import pandas as pd
import json
from PyPDF2 import PdfReader

# --- Load and process DRVs CSV ---
print("DRVs_Adults.csv")
df = pd.read_csv("data/DRVs_Adults.csv")

# Clean column names: lowercase, strip spaces, replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Print column names and preview first 10 rows
print("Columns:", df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10))

# Print missing values per column
print("\nMissing values:")
print(df.isnull().sum())

# Save cleaned DRVs dataframe to JSON for later use
df.to_json("data/DRVs_Adults.json", orient="records", indent=2)


# --- Load and process Nutritional Database CSV ---
print("\n\nNutritional_Database_Ciqual.csv")
ciqual = pd.read_csv("data/Nutritional_Database_Ciqual.csv", sep=",")

# Clean column names
ciqual.columns = ciqual.columns.str.strip().str.lower().str.replace(" ", "_")

# Print column names and preview first 10 rows
print("Columns:", ciqual.columns.tolist())
print("\nFirst 10 rows:")
print(ciqual.head(10))

# Print missing values per column
print("\nMissing values:")
print(ciqual.isnull().sum())

# Save cleaned Ciqual dataframe to JSON for later use
ciqual.to_json("data/Nutritional_Database_Ciqual.json", orient="records", indent=2)


# --- PDF Preview Function and Saving ---
def preview_pdf(path, num_pages=3):
    """
    Reads and previews text from the first `num_pages` of a PDF file.
    Cleans extra whitespace. Returns a list of strings (page texts).
    """
    try:
        print(f"\nPreviewing: {path}")
        reader = PdfReader(path)
        pages_text = []
        for i, page in enumerate(reader.pages[:num_pages]):
            text = page.extract_text()
            if text:
                cleaned_text = ' '.join(text.split())  # Collapse all whitespace
            else:
                cleaned_text = ""
            pages_text.append(cleaned_text)
            print(f"\n--- Page {i + 1} ---")
            print(cleaned_text[:1000])  # Show first 1000 characters
        return pages_text
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

# Read and save contents of Dietary Sources PDF
dietary_pages = preview_pdf("data/Dietary_Sources.pdf")
with open("data/Dietary_Sources.json", "w") as f:
    json.dump({"pages": dietary_pages}, f, indent=2)

# Read and save contents of Medication Description PDF
medication_pages = preview_pdf("data/Medication_Description.pdf")
with open("data/Medication_Description.json", "w") as f:
    json.dump({"pages": medication_pages}, f, indent=2)