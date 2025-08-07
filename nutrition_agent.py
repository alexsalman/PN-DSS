# nutrition agent that creates nutrition plans for patients
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess
import token_1

pat = 20189
r= token_1.get_patient_data(pat)
# print('HTTP Status: ' + str(r.status_code))
print(r[1]['centro'])

# # Load CSV
# df = pd.read_csv("patients.csv")

# # Load embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Create embeddings
# texts = df.apply(lambda row: f"{row.name}, {row.age}yo, {row.gender}, {row.weight}kg, {row.height}cm, Goal: {row.goal}, Conditions: {row.medical_conditions}, Prefs: {row.preferences}", axis=1)
# embeddings = model.encode(texts.tolist(), convert_to_numpy=True)

# # Setup FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)

# # Prompt user
# query = input("Enter patient name to generate a nutrition plan: ").strip()
# patient_idx = df.index[df['name'].str.lower() == query.lower()].tolist()

# if not patient_idx:
#     print("Patient not found.")
#     exit()

# # Retrieve patient info
# i = patient_idx[0]
# context = texts[i]

# # Build LLM prompt
# prompt = f"""
# You are a nutritionist. Create a personalized weekly nutrition plan for the following patient:

# {context}

# Make it specific, medically appropriate, and consider preferences.
# """

# # Call local LLM (Ollama)
# print("\nGenerating plan with Mistral...\n")
# result = subprocess.run(
#     ["ollama", "run", "mistral"],
#     input=prompt.encode(),
#     stdout=subprocess.PIPE
# )

# print(result.stdout.decode())