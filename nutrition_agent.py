# nutrition_agent.py

import subprocess
import re
from agent_core import NutritionAgentCore
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    ListFlowable, ListItem, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import token_1


def format_model_output(output_text, styles):
    flowables = []

    # --- Split reasoning ("thinking") before Day 1 ---
    thinking_split = re.split(r"(?i)\b(day\s*1\b[:\-]?)", output_text, maxsplit=1)
    if len(thinking_split) == 3:
        thinking, day1_label, rest = thinking_split
        flowables.append(Paragraph("💭 Model Reasoning", styles['Heading3']))
        flowables.append(Spacer(1, 6))
        flowables.append(Paragraph(thinking.strip().replace("\n", "<br/>"), styles['BodyText']))
        flowables.append(Spacer(1, 12))
        output_text = f"{day1_label}{rest}"  # resume days only

    # --- Split by "Day 1", "Day 2", etc. ---
    day_chunks = re.split(r"(?i)(day\s*\d+\b[:\-]?)", output_text)
    mono = ParagraphStyle(
        name='Mono',
        fontName='Courier',
        fontSize=9,
        leading=11,
        leftIndent=20
    )

    for i in range(1, len(day_chunks), 2):
        day_title = day_chunks[i].strip().capitalize()
        day_content = day_chunks[i + 1].strip()

        flowables.append(Paragraph(f"🍽️ {day_title}", styles['Heading3']))
        flowables.append(Spacer(1, 6))

        meal_lines = [line.strip() for line in day_content.split('\n') if line.strip()]
        list_items = []

        for line in meal_lines:
            # Highlight meal headers (e.g. Breakfast:, Snack:)
            if re.match(r"^(breakfast|lunch|dinner|snack)", line.lower()):
                list_items.append(ListItem(Paragraph(f"<b>{line}</b>", styles['Normal']), leftIndent=10))
            # Indented plain or tabular lines
            elif ":" in line or "-" in line:
                list_items.append(ListItem(Paragraph(line, mono), leftIndent=18))
            else:
                list_items.append(ListItem(Preformatted(line, mono), leftIndent=18))

        flowables.append(ListFlowable(list_items, bulletType='bullet'))
        flowables.append(Spacer(1, 12))

    return flowables


def save_to_pdf(filename, prompt, model_output):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=60,
        bottomMargin=60
    )

    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 13

    flowables = []

    # Prompt Section
    flowables.append(Paragraph("🧠 Prompt Used", styles['Heading2']))
    flowables.append(Spacer(1, 6))
    flowables.append(Paragraph(prompt.replace("\n", "<br/>"), styles['BodyText']))
    flowables.append(PageBreak())

    # Model Output
    flowables.append(Paragraph("📋 Model Output", styles['Heading2']))
    flowables.append(Spacer(1, 12))
    flowables.extend(format_model_output(model_output, styles))

    doc.build(flowables)


# === Main logic ===

# Step 1: Load patient
pat_id = 50834
r = token_1.get_patient_data(pat_id)
patient = r[0]

# Step 2: Run evaluation
agent = NutritionAgentCore()
report = agent.evaluate_patient(patient)

# Step 3: Build prompt
if "error" in report:
    context = f"Patient data could not be evaluated: {report['error']}"
else:
    context = f"""
Patient Profile:
- Age: {report['age']} ({report['gender']})
- BMI: {report['bmi']} ({report['bmi_class']})
- Waist/Hip Ratio: {report['waist_hip_ratio']} ({report['waist_hip_status']})
- BMR: {report['bmr']} kcal
- TEE: {report['tee']} kcal
- Energy intake: {report['energy_status']}
- Protein intake: {report['protein_status']}
- Fat intake: {report['fat_status']}
- Fiber intake: {report['fiber_status']}
- Mediterranean diet adherence: {report['mediterranean_adherence']}
- Firmicutes/Bacteroidota ratio: {report['f_b_ratio']} ({report['f_b_status']})
- Diagnosed conditions: {', '.join(report['diseases']) if report['diseases'] else 'None'}
"""

prompt = f"""
You are a licensed nutritionist. Based on the following clinical profile, create a personalized 3-day nutrition plan:

{context}

Each day should include 3 meals and 2 snacks, adjusted for the patient's estimated energy needs and any diagnosed conditions. Use medically sound principles and include specific food examples. Think step-by-step before giving the plan.
"""

# Step 4: Run local model
print("\nGenerating nutrition plan using local Ollama model...\n")
result = subprocess.run(
    ["ollama", "run", "gpt-oss:20b"],
    input=prompt.encode(),
    stdout=subprocess.PIPE
)
output = result.stdout.decode()

# Step 5: Save PDF
save_to_pdf("nutrition_report.pdf", prompt, output)
print("\n✅ PDF saved to nutrition_report.pdf\n")

# # nutrition_agent.py
# # nutrition agent that creates nutrition plans for patients
# import pandas as pd
# import faiss
# import numpy as np
# import subprocess
# from sentence_transformers import SentenceTransformer
# import token_1

# # ---- Get patient info from REDCap ----
# pat_id = '20189'  # Replace with actual ID or input()
# #50834
# r = token_1.get_patient_data(pat_id)
# patient = r[0]

# # Basic fields
# age = patient['edad_v22']
# sex = patient['sexo']
# goal = "Diabetes control" if patient['diabetico_v22'] == '1' else "General wellness"

# # ---- Optional CSV for example patient embeddings ----
# try:
#     df = pd.read_csv("test.csv")

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     texts = df.apply(lambda row: f"{row.name}, {row.age}yo, {row.gender}, {row.weight}kg, {row.height}cm, Goal: {row.goal}, Conditions: {row.medical_conditions}, Prefs: {row.preferences}", axis=1)
#     embeddings = model.encode(texts.tolist(), convert_to_numpy=True)

#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)

#     query = input("Enter patient name to generate a nutrition plan: ").strip()
#     match_idx = df.index[df['name'].str.lower() == query.lower()].tolist()

#     if match_idx:
#         context = texts[match_idx[0]]
#     else:
#         context = f"Patient: {age}yo, sex: {sex}, Goal: {goal}"
# except Exception:
#     # Fallback to just the REDCap data
#     context = f"""Patient info:
#     - Age: {age}
#     - Sex: {sex}
#     - Diabetic: {patient.get('diabetico_v22')}
#     - Weight: {patient.get('peso_v22')} kg
#     - Height: {patient.get('talla_v22')} cm
#     - Fiber intake: {patient.get('dietary_fibre_g')} g
#     - Total kcal: {patient.get('total_energy_kcal')}
#     - Macronutrients: carbs {patient.get('hc_g')}g, protein {patient.get('proteins_g')}g, fat {patient.get('fat_g')}g
#     - Microbiome: Firmicutes {patient.get('p_firmicutes')}%, Bacteroidota {patient.get('p_bacteroidota')}%
#     """

# # ---- Prompt to Ollama ----
# prompt = f"""
# You are a licensed nutritionist. Based on the following clinical and dietary profile, create a personalized 3-day nutrition plan:

# {context}

# Ensure the plan is medically safe, includes meal examples, and follows current nutritional guidelines.
# """

# print("\nGenerating nutrition plan using local Ollama model...\n")
# result = subprocess.run(
#     ["ollama", "run", "gpt-oss:20b"], 
#     input=prompt.encode(),
#     stdout=subprocess.PIPE
# )

# print(result.stdout.decode())