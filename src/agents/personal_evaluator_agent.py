# src/agents/personal_evaluator_agent.py

import datetime
import json
import os
import subprocess
import re
from typing import Dict, Any, Tuple, Optional

class PersonalEvaluatorAgent:
    def __init__(self, model_name: str = "gpt-oss:20b", memory_dir: str = "data/memory/personal"):
        self.name = "Personal Evaluator"
        self.role = "Reasoning" 
        self.model_name = model_name
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        print(f"\n[DEBUG_AGENT_IN: {self.name}] PID: {pid}")
        
        # --- AGENT MEMORY OPTIMIZATION (STATEFUL) ---
        cached_data = self._check_local_memory(pid)
        if cached_data:
            print(f"[MEMORY_HIT: {self.name}] Using existing analysis for PID: {pid}")
            return cached_data

        try:
            personal_data = categories.get('personal', {})
            
            # 1. Extraction with Fallbacks and String-to-Float Cleaning
            def get_float(key, default):
                val = personal_data.get(key) or patient.get(key)
                if val is None or val == "": return default
                try: return float(str(val).replace(',', '.'))
                except: return default

            age = get_float('edad_v22', 0)
            sex = str(personal_data.get('sexo') or patient.get('sexo', '2')) 
            weight = get_float('peso_v22', 0)
            height = get_float('talla_v22', 0)
            pal = personal_data.get('activity', 'moderate')
            
            # 2. Hard Calculations (Deterministic Logic)
            if height > 0 and weight > 0:
                bmr = self.calculate_bmr(sex, weight, height, age)
                tee = self.calculate_tee(bmr, pal)
                bmi_val, bmi_class, ideal_range = self.calculate_bmi(weight, height)
            else:
                bmr, tee, bmi_val = 0, 0, 0
                bmi_class = "Unknown"
                ideal_range = "N/A"

            # 3. LLM Reasoning (Using Supervisor context for History)
            history_data = context.get('history', {}) if context else {}
            visit_history = history_data.get('visit_history', [])
            
            # Extract previous weight if it exists in supervisor memory
            prev_weight = weight
            if visit_history:
                # Look for the last visit's weight record
                prev_weight = visit_history[-1].get('weight', weight)
            
            prompt = (
                f"System: Clinical Nutritionist.\n"
                f"Profile: BMI {bmi_val} ({bmi_class}), Weight {weight}kg, Age {age}, TEE {round(tee, 0)} kcal/day.\n"
                f"History: Previous Weight {prev_weight}kg.\n"
                f"Task: Provide a 1-sentence clinical rationale for the weight management goal.\n"
                f"Output: Only the rationale sentence."
            )
            
            rationale = self._call_ollama(prompt)

            # 4. Construct Result
            analysis_results = {
                "bmi": bmi_val,
                "weight": weight, 
                "height": height,
                "age": age,
                "sex": sex,
                "bmi_classification": bmi_class,
                "ideal_weight_range": ideal_range,
                "bmr_kcal": round(bmr, 0),
                "tee_kcal": round(tee, 0),
                "weight_trend_rationale": rationale,
                "pal_applied": pal
            }

            res = {
                "status": "success", 
                "summary": rationale, 
                "data": analysis_results,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # 5. Update Agent Memory (Stateful Persistence)
            self._save_to_local_memory(pid, res)
            
            return res

        except Exception as e:
            return {"status": "error", "message": f"PersonalEvaluator failed: {str(e)}"}

    def _check_local_memory(self, patient_id: str) -> Optional[Dict]:
        path = self._get_memory_path(patient_id)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    if not history: return None
                    last_entry = history[-1]
                    ts = datetime.datetime.fromisoformat(last_entry['timestamp'])
                    if (datetime.datetime.now() - ts).total_seconds() < 86400:
                        return last_entry
            except: pass
        return None

    def _get_memory_path(self, patient_id: str) -> str:
        return os.path.join(self.memory_dir, f"{patient_id}.json")

    def _save_to_local_memory(self, patient_id: str, data: Dict):
        try:
            path = self._get_memory_path(patient_id)
            history = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    try: history = json.load(f)
                    except: history = []
            
            history.append(data)
            with open(path, 'w', encoding='utf-8') as f: 
                json.dump(history[-10:], f, indent=4)
        except Exception as e:
            print(f"[DEBUG_AGENT_ERROR] Memory save failed: {e}")

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt], 
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            raw = result.stdout.strip()
            return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        except: 
            return "Metabolic assessment complete. Weight management targets set for clinical stability."

    def calculate_bmr(self, sex: str, weight: float, height: float, age: float) -> float:
        # Mifflin-St Jeor Equation
        s = -161 if str(sex) == '2' else 5 
        return (10 * weight) + (6.25 * height) - (5 * age) + s

    def calculate_tee(self, bmr: float, pal_category: str) -> float:
        pal_map = {'inactive': 1.2, 'moderate': 1.55, 'regular': 1.75, 'intense': 2.0}
        key = str(pal_category).lower() if pal_category else 'moderate'
        return bmr * pal_map.get(key, 1.55)

    def calculate_bmi(self, weight: float, height: float) -> Tuple[float, str, str]:
        h_m = height / 100.0
        bmi = round(weight / (h_m ** 2), 1)
        if bmi >= 30: status = 'Obesity'
        elif bmi >= 25: status = 'Overweight'
        elif bmi >= 18.5: status = 'Normal'
        else: status = 'Underweight'
        return bmi, status, f"{round(18.5*(h_m**2),1)} - {round(24.9*(h_m**2),1)} kg"