# src/agents/personal_evaluator_agent.py

import datetime
import subprocess
import re
import json
import os
from typing import Dict, Any, Tuple, Optional

class PersonalEvaluatorAgent:
    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.name = "Personal Evaluator"
        self.role = "Reasoning" 
        self.model_name = model_name
        self.memory_dir = "memory"

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        memory_path = f"{self.memory_dir}/{pid}.json"
        
        try:
            # 1. INDEPENDENT READ
            historical_rationale = "No previous history."
            if os.path.exists(memory_path):
                with open(memory_path, 'r', encoding='utf-8') as f:
                    full_memory = json.load(f)
                    historical_rationale = full_memory.get("agent_memories", {}).get("personal_evaluator_memory", "No previous history.")

            personal_data = categories.get('personal', {})
            
            # Extraction with Corrected Logic
            def get_float(key, default):
                val = personal_data.get(key) or patient.get(key)
                if val is None or val == "": return default
                try: return float(str(val).replace(',', '.'))
                except: return default

            age = get_float('edad_v22', 30)
            sex = str(personal_data.get('sexo') or patient.get('sexo', '2')) 
            weight = get_float('peso_v22', 0)
            height = get_float('talla_v22', 0)
            pal = personal_data.get('activity', 'moderate')
            
            # 2. DETERMINISTIC CALCULATIONS
            if height > 10 and weight > 10: # Ensure valid clinical inputs
                bmr = self.calculate_bmr(sex, weight, height, age)
                tee = self.calculate_tee(bmr, pal)
                bmi_val, bmi_class, ideal_range = self.calculate_bmi(weight, height)
            else:
                bmr, tee, bmi_val = 0, 0, 0
                bmi_class = "Data Missing"
                ideal_range = "N/A"

            # 3. LLM Reasoning
            prompt = (
                f"System: Clinical Nutritionist Analyst.\n"
                f"Profile: BMI {bmi_val} ({bmi_class}), Weight {weight}kg, Age {age}.\n"
                f"History: {historical_rationale}\n"
                f"Task: Provide a 1-sentence clinical rationale for the BMI classification and weight target.\n"
                f"Constraint: No intro, no thinking, no artifacts."
            )
            
            new_rationale = self._call_ollama(prompt)

            analysis_results = {
                "bmi": bmi_val,
                "weight": weight, 
                "height": height,
                "age": age,
                "sex": sex,
                "bmi_classification": bmi_class,
                "ideal_weight_range": ideal_range,
                "bmr_kcal": int(round(bmr, 0)),
                "tee_kcal": int(round(tee, 0)),
                "pal_applied": pal
            }

            # 4. BLACKBOARD WRITE
            self._update_memory(memory_path, new_rationale, analysis_results)

            return {
                "status": "success", 
                "summary": new_rationale, 
                "data": analysis_results
            }

        except Exception as e:
            return {"status": "error", "message": f"PersonalEvaluator failed: {str(e)}"}

    def _update_memory(self, path: str, summary: str, data_block: Dict):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.setdefault("agent_memories", {})
            data["agent_memories"]["personal_evaluator_memory"] = summary
            data["agent_memories"]["personal_evaluator_data"] = data_block
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(["ollama", "run", self.model_name, prompt], 
                                 capture_output=True, text=True, check=True, encoding='utf-8')
            output = result.stdout.strip()
            # Wipe all thought/thinking variations
            output = re.sub(r'<(think|thought)>.*?</\1>', '', output, flags=re.DOTALL | re.IGNORECASE)
            output = re.sub(r'(?i)Thinking\.\.\..*?(\.\.\.done thinking|Let\'s think).*?\.?', '', output, flags=re.DOTALL)
            return output.strip()
        except: 
            return "Metabolic assessment complete."

    def calculate_bmr(self, sex: str, weight: float, height: float, age: float) -> float:
        """Mifflin-St Jeor Equation."""
        # Male: 1 or 'male'; Female: 2 or 'female'
        s = 5 if str(sex) in ['1', 'male', 'M'] else -161
        return (10 * weight) + (6.25 * height) - (5 * age) + s

    def calculate_tee(self, bmr: float, pal_category: str) -> float:
        pal_map = {'inactive': 1.2, 'moderate': 1.55, 'regular': 1.75, 'intense': 2.0}
        factor = pal_map.get(str(pal_category).lower(), 1.55)
        return bmr * factor

    def calculate_bmi(self, weight: float, height: float) -> Tuple[float, str, str]:
        h_m = height / 100.0
        if h_m <= 0: return 0, "Error", "N/A"
        bmi = round(weight / (h_m ** 2), 1)
        if bmi >= 30: status = 'Obesity'
        elif bmi >= 25: status = 'Overweight'
        elif bmi >= 18.5: status = 'Normal'
        else: status = 'Underweight'
        return bmi, status, f"{round(18.5*(h_m**2),1)} - {round(24.9*(h_m**2),1)} kg"
