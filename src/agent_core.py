import math
import numpy as np
import pandas as pd


class NutritionAgentCore:
    def __init__(self, drv_path='DRVs_Adults.csv'):
        self.drv = pd.read_csv(drv_path)
        self.drv.set_index('Nutrient', inplace=True)

    # --- Core Calculations ---
    def calculate_bmr(self, sex, weight, height, age):
        if sex == 'male' or str(sex) == '2':
            return (10 * weight) + (6.25 * height) - (5 * age) + 5
        elif sex == 'female' or str(sex) == '1':
            return (10 * weight) + (6.25 * height) - (5 * age) - 161
        return None

    def calculate_tee(self, bmr, pal_category):
        pal_map = {'inactive': 1.4, 'moderate': 1.6, 'regular': 1.8, 'intense': 2.0}
        return bmr * pal_map.get(pal_category.lower(), 1.6)

    def calculate_bmi(self, weight, height):
        h_m = height / 100
        return round(weight / (h_m ** 2), 1)

    def classify_bmi(self, bmi):
        if bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal'
        elif bmi < 30: return 'Overweight'
        return 'Obesity'

    def waist_hip_ratio(self, waist, hip, sex):
        if not waist or not hip: return None, None
        ratio = waist / hip
        threshold = 0.93 if str(sex) == '2' else 0.85
        return round(ratio, 2), ('Elevated' if ratio > threshold else 'Normal')

    # --- Nutritional Rules ---
    def classify_energy_intake(self, current_kcal, tee):
        if current_kcal is None or tee is None: return 'Unknown'
        if current_kcal < tee: return 'Under intake'
        elif current_kcal > tee: return 'Over intake'
        return 'Adequate'

    def classify_protein(self, current_prot, weight):
        required = 0.83 * weight
        if current_prot < required: return 'Under intake'
        elif current_prot > required * 1.2: return 'Over intake'
        return 'Adequate'

    def classify_fat_percent(self, fat_g, kcal):
        fat_pct = (fat_g * 9 / kcal) * 100 if kcal else 0
        if fat_pct < 20: return 'Under intake'
        elif fat_pct <= 35: return 'Normal'
        return 'Over intake'

    def classify_fiber(self, fiber_g):
        if 25 <= fiber_g <= 30: return 'Normal'
        elif fiber_g < 25: return 'Under intake'
        return 'Over intake'

    def classify_mediterranean_score(self, score):
        if score <= 7: return 'Low adherence'
        elif score <= 9: return 'Medium adherence'
        return 'High adherence'

    def classify_fb_ratio(self, ratio):
        if ratio < 1.57: return 'Low'
        elif ratio <= 1.99: return 'Normal'
        return 'High'

    def classify_micronutrients(self, patient_data, sex):
        micronutrient_status = {}
        missing_fields = []

        sex_label = 'Male' if str(sex) == '2' else 'Female'
        subset = self.drv[[sex_label]].dropna()

        for nutrient, row in subset.iterrows():
            field = row.name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "").replace("(", "").replace(")", "")
            val = patient_data.get(field)
            if val in [None, '', 0]:
                missing_fields.append(field)
                continue
            try:
                val = float(val)
                ref = float(row[sex_label])
                low, high = ref * 0.8, ref * 1.2
                if val < low:
                    status = "Under intake"
                elif val > high:
                    status = "Over intake"
                else:
                    status = "Adequate"
                micronutrient_status[nutrient] = {
                    "value": round(val, 2),
                    "recommended": ref,
                    "status": status
                }
            except Exception:
                missing_fields.append(field)

        return micronutrient_status, missing_fields

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

        micronutrient_status, missing_micro = self.classify_micronutrients(p, sex)

        diseases = [
            name for name, cond in [
                ('Diabetes', p.get('diabetico_v22') == '1'),
                ('Hypertension', p.get('hta') == '1'),
                ('High cholesterol', p.get('cholesterol') == '1'),
                ('Cardiovascular disease', p.get('cvd') == '1'),
                ('Inflammatory bowel disease', p.get('ibd') == '1')
            ] if cond
        ]

        basic_missing = {
            k: p.get(k) in [None, '', 0] for k in [
                'edad_v22', 'sexo', 'peso_v22', 'talla_v22',
                'waist', 'hip', 'activity',
                'total_energy_kcal', 'proteins_g', 'fat_g', 'dietary_fibre_g'
            ]
        }

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
            "diseases": diseases,
            "micronutrient_status": micronutrient_status,
            "missing": {**basic_missing, **{k: True for k in missing_micro}}
        }