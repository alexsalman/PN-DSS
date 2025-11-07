# agent_core.py
import math
import numpy as np


class NutritionAgentCore:
    def __init__(self):
        pass

    # --- Core Calculations ---
    def calculate_bmr(self, sex, weight, height, age):
        """Calculate Basal Metabolic Rate (Mifflin-St Jeor Equation)."""
        if sex == 'male' or str(sex) == '2':
            return (10 * weight) + (6.25 * height) - (5 * age) + 5
        elif sex == 'female' or str(sex) == '1':
            return (10 * weight) + (6.25 * height) - (5 * age) - 161
        else:
            return None

    def calculate_tee(self, bmr, pal_category):
        pal_map = {
            'inactive': 1.4,
            'moderate': 1.6,
            'regular': 1.8,
            'intense': 2.0
        }
        pal = pal_map.get(pal_category.lower(), 1.6)
        return bmr * pal

    def calculate_bmi(self, weight, height):
        """Height in cm."""
        h_m = height / 100
        return round(weight / (h_m ** 2), 1)

    def classify_bmi(self, bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Normal'
        elif 25 <= bmi < 30:
            return 'Overweight'
        else:
            return 'Obesity'

    def waist_hip_ratio(self, waist, hip, sex):
        if not waist or not hip:
            return None, None
        ratio = waist / hip
        threshold = 0.93 if str(sex) == '2' else 0.85
        status = 'Elevated' if ratio > threshold else 'Normal'
        return round(ratio, 2), status

    # --- Nutritional Rules ---
    def classify_energy_intake(self, current_kcal, tee):
        if current_kcal is None or tee is None:
            return 'Unknown'
        if current_kcal < tee:
            return 'Under intake'
        elif current_kcal > tee:
            return 'Over intake'
        else:
            return 'Adequate'

    def classify_protein(self, current_prot, weight):
        required = 0.83 * weight
        if current_prot < required:
            return 'Under intake'
        elif current_prot > required * 1.2:
            return 'Over intake'
        else:
            return 'Adequate'

    def classify_fat_percent(self, fat_g, kcal):
        fat_pct = (fat_g * 9 / kcal) * 100 if kcal else 0
        if fat_pct < 20:
            return 'Under intake'
        elif 20 <= fat_pct <= 35:
            return 'Normal'
        else:
            return 'Over intake'

    def classify_fiber(self, fiber_g):
        if 25 <= fiber_g <= 30:
            return 'Normal'
        elif fiber_g < 25:
            return 'Under intake'
        else:
            return 'Over intake'

    def classify_mediterranean_score(self, score):
        if score <= 7:
            return 'Low adherence'
        elif 8 <= score <= 9:
            return 'Medium adherence'
        else:
            return 'High adherence'

    def classify_fb_ratio(self, ratio):
        if ratio < 1.57:
            return 'Low'
        elif 1.57 <= ratio <= 1.99:
            return 'Normal'
        else:
            return 'High'

    # --- Evaluation Wrapper ---
    def evaluate_patient(self, p):
        try:
            age = float(p.get('edad_v22', 0))
            sex = p.get('sexo', '2')
            weight = float(p.get('peso_v22', 0))
            height = float(p.get('talla_v22', 0))
            waist = float(p.get('waist', 0) or 0)
            hip = float(p.get('hip', 0) or 0)
            kcal = float(p.get('total_energy_kcal', 0))
            protein = float(p.get('proteins_g', 0))
            fat = float(p.get('fat_g', 0))
            fiber = float(p.get('dietary_fibre_g', 0))
            fb_ratio = float(p.get('p_firmicutes', 0)) / float(p.get('p_bacteroidota', 1))
        except Exception:
            return {"error": "Invalid patient data."}

        # --- Compute ---
        bmr = self.calculate_bmr(sex, weight, height, age)
        tee = self.calculate_tee(bmr, p.get('activity', 'moderate'))
        bmi = self.calculate_bmi(weight, height)
        bmi_class = self.classify_bmi(bmi)
        wh_ratio, wh_status = self.waist_hip_ratio(waist, hip, sex)
        energy_status = self.classify_energy_intake(kcal, tee)
        protein_status = self.classify_protein(protein, weight)
        fat_status = self.classify_fat_percent(fat, kcal)
        fiber_status = self.classify_fiber(fiber)
        med_score = float(p.get('med_diet_score', 0))
        med_status = self.classify_mediterranean_score(med_score)
        fb_status = self.classify_fb_ratio(fb_ratio)

        diseases = []
        if p.get('diabetico_v22') == '1': diseases.append('Diabetes')
        if p.get('hta') == '1': diseases.append('Hypertension')
        if p.get('cholesterol') == '1': diseases.append('High cholesterol')
        if p.get('cvd') == '1': diseases.append('Cardiovascular disease')
        if p.get('ibd') == '1': diseases.append('Inflammatory bowel disease')

        return {
            "id": p.get('record_id', 'unknown'),
            "gender": "Male" if sex == '2' else "Female",
            "age": age,
            "bmr": round(bmr, 2),
            "tee": round(tee, 2),
            "bmi": bmi,
            "bmi_class": bmi_class,
            "waist_hip_ratio": wh_ratio,
            "waist_hip_status": wh_status,
            "energy_status": energy_status,
            "protein_status": protein_status,
            "fat_status": fat_status,
            "fiber_status": fiber_status,
            "mediterranean_adherence": med_status,
            "f_b_ratio": round(fb_ratio, 2),
            "f_b_status": fb_status,
            "diseases": diseases
        }