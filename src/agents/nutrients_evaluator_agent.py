# src/agents/nutrients_evaluator_agent.py

import json
import os
import datetime
import subprocess
import re
from typing import Any, Dict, Optional, List

class NutrientsEvaluatorAgent:
    def __init__(self, 
                 model_name: str = "gpt-oss:20b",
                 nutrients_db_path='data/Nutritional_Database_Ciqual.json'):
        self.name = "Nutrients Evaluator"
        self.role = "Reasoning"
        self.model_name = model_name
        self.memory_dir = "memory" 
        self.ciqual_db = self._load_db(nutrients_db_path)

    def _load_db(self, path: str) -> List[Dict]:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception: return []
        return []

    def _clean_val(self, val: Any) -> float:
        if val in [None, "-", ""]: return 0.0
        try:
            s_val = str(val).replace('<', '').replace('>', '').replace(',', '.').strip()
            return float(s_val)
        except (ValueError, TypeError): return 0.0

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        memory_path = f"{self.memory_dir}/{pid}.json"

        try:
            # 1. INDEPENDENT READ
            p_eval_data = {}
            if os.path.exists(memory_path):
                with open(memory_path, 'r', encoding='utf-8') as f:
                    full_memory = json.load(f)
                    p_eval_data = full_memory.get("agent_memories", {}).get("personal_evaluator_data", {})

            # 2. Extract Values
            weight = float(p_eval_data.get('weight') or patient.get('peso_v22') or 70.0)
            bmi = float(p_eval_data.get('bmi') or 25.0)
            # CRITICAL: Pull TEE from Personal Evaluator to anchor the Meal Planner
            energy_target = float(p_eval_data.get('tee_kcal') or 2000.0)
            
            raw_nutrients = categories.get('nutrients', {})

            # 3. Deterministic Targets
            protein_target = self.calculate_protein_needs(weight, bmi)
            fiber_target = 30.0 
            actual_protein = self._clean_val(raw_nutrients.get('proteins_g', 0))

            # 4. LLM Synthesis (Layer 3: Clinical Rationale)
            prompt = (
                f"System: You are a Clinical Dietitian. Provide ONLY the final rationale.\n"
                f"Context: Patient Weight {weight}kg, BMI {bmi}, Target {protein_target}g protein.\n"
                f"Task: Write a 1-sentence clinical rationale for this protein target.\n"
                f"Constraint: No 'Thinking...', no intro, no conversational filler."
            )
            
            raw_rationale = self._call_ollama(prompt)
            new_rationale = self._sanitize_rationale(raw_rationale)

            # 5. Build Result Data Payload (Ensuring Energy is passed to Planner)
            analysis_data = {
                "energy_target_kcal": energy_target,
                "protein_target_g": protein_target,
                "fiber_target_g": fiber_target,
                "actual_protein_g": actual_protein,
                "clinical_rationale": new_rationale
            }

            # 6. INDEPENDENT WRITE
            self._update_memory(memory_path, new_rationale, analysis_data)

            return {"status": "success", "summary": new_rationale, "data": analysis_data}

        except Exception as e:
            return {"status": "error", "message": f"NutrientsEvaluator failed: {str(e)}"}

    def _sanitize_rationale(self, text: str) -> str:
        """Purges internal instructions and LLM self-talk."""
        # Remove common instruction leaks found in your terminal logs
        text = re.sub(r'(?i)Thinking\.\.\..*?(\.\.\.done thinking|Let\'s think).*?\.?', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)^(We need to|Provide a|The rationale|Let\'s).*?(:|\.)', '', text).strip()
        return text or "Protein target adjusted for clinical requirements."

    def calculate_protein_needs(self, weight: float, bmi: float) -> float:
        """Updated clinical multiplier logic to prevent target drift."""
        if bmi >= 30:
            multiplier = 1.2  # Obese: Muscle preservation
        elif bmi >= 25:
            multiplier = 1.0  # Overweight: Maintenance
        else:
            multiplier = 0.8  # Normal: Baseline
        return round(float(weight) * multiplier, 1)

    def _update_memory(self, path: str, summary: str, data: Dict):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                memory_json = json.load(f)
            if "agent_memories" not in memory_json:
                memory_json["agent_memories"] = {}
            memory_json["agent_memories"]["nutrients_evaluator_memory"] = summary
            memory_json["agent_memories"]["nutrients_evaluator_data"] = data
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(memory_json, f, indent=4)

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(["ollama", "run", self.model_name, prompt], 
                                 capture_output=True, text=True, check=True, encoding='utf-8')
            output = result.stdout.strip()
            return re.sub(r'<(think|thought)>.*?</\1>', '', output, flags=re.DOTALL | re.IGNORECASE).strip()
        except: return "Targets adjusted for clinical BMI status."