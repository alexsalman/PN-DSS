# nutrition agent that creates nutrition plans for patients
import pandas as pd
import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer
import token_1

# ---- Get patient info from REDCap ----
pat_id = '20189'  # Replace with actual ID or input()
r = token_1.get_patient_data(pat_id)
patient = r[0]

# Basic fields
age = patient['edad_v22']
sex = patient['sexo']
goal = "Diabetes control" if patient['diabetico_v22'] == '1' else "General wellness"

# ---- Optional CSV for example patient embeddings ----
try:
    df = pd.read_csv("test.csv")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df.apply(lambda row: f"{row.name}, {row.age}yo, {row.gender}, {row.weight}kg, {row.height}cm, Goal: {row.goal}, Conditions: {row.medical_conditions}, Prefs: {row.preferences}", axis=1)
    embeddings = model.encode(texts.tolist(), convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = input("Enter patient name to generate a nutrition plan: ").strip()
    match_idx = df.index[df['name'].str.lower() == query.lower()].tolist()

    if match_idx:
        context = texts[match_idx[0]]
    else:
        context = f"Patient: {age}yo, sex: {sex}, Goal: {goal}"
except Exception:
    # Fallback to just the REDCap data
    context = f"""Patient info:
    - Age: {age}
    - Sex: {sex}
    - Diabetic: {patient.get('diabetico_v22')}
    - Weight: {patient.get('peso_v22')} kg
    - Height: {patient.get('talla_v22')} cm
    - Fiber intake: {patient.get('dietary_fibre_g')} g
    - Total kcal: {patient.get('total_energy_kcal')}
    - Macronutrients: carbs {patient.get('hc_g')}g, protein {patient.get('proteins_g')}g, fat {patient.get('fat_g')}g
    - Microbiome: Firmicutes {patient.get('p_firmicutes')}%, Bacteroidota {patient.get('p_bacteroidota')}%
    """

# ---- Prompt to Ollama ----
prompt = f"""
You are a licensed nutritionist. Based on the following clinical and dietary profile, create a personalized 3-day nutrition plan:

{context}

Ensure the plan is medically safe, includes meal examples, and follows current nutritional guidelines.
"""

print("\nGenerating nutrition plan using local Ollama model...\n")
result = subprocess.run(
    ["ollama", "run", "gpt-oss:20b"], 
    input=prompt.encode(),
    stdout=subprocess.PIPE
)

print(result.stdout.decode())