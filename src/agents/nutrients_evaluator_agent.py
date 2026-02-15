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
                 nutrients_db_path='data/Nutritional_Database_Ciqual.json',
                 memory_dir: str = "data/memory/nutrients"):
        self.name = "Nutrients Evaluator"
        self.role = "Reasoning"
        self.model_name = model_name
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Load CIQUAL Database
        self.ciqual_db = self._load_db(nutrients_db_path)

    def _load_db(self, path: str) -> List[Dict]:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not load Ciqual DB: {e}")
        return []

    def _clean_val(self, val: Any) -> float:
        """Parses Ciqual string values like '45,4' or '< 0,3' into floats."""
        if val is None or val == "-" or val == "":
            return 0.0
        try:
            # Handle clinical string formats found in CIQUAL
            s_val = str(val).replace('<', '').replace('>', '').replace(',', '.').replace('[', '').replace(']', '').strip()
            return float(s_val)
        except (ValueError, TypeError):
            return 0.0

    def find_food_nutrients(self, query: str) -> List[Dict]:
        """Search tool to find food items in Ciqual by name (for future trace expansion)."""
        query = query.lower()
        results = []
        for item in self.ciqual_db:
            name = item.get('alim_nom_eng', '')
            if name and query in name.lower():
                results.append({
                    "name": name,
                    "protein": self._clean_val(item.get('protein_(g/100g)')),
                    "carbs": self._clean_val(item.get('carbohydrate_(g/100g)')),
                    "fat": self._clean_val(item.get('fat_(g/100g)')),
                    "fiber": self._clean_val(item.get('fibres_(g/100g)'))
                })
        return results[:5]

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        print(f"\n[DEBUG_AGENT_IN: {self.name}] PID: {pid}")
        
        # --- AGENT MEMORY OPTIMIZATION (STATEFUL) ---
        cached_data = self._check_local_memory(pid)
        if cached_data:
            print(f"[MEMORY_HIT: {self.name}] Using existing analysis for PID: {pid}")
            return cached_data

        try:
            # 1. State Retrieval from Personal Evaluator
            p_eval = context.get('personal_evaluator', {}).get('data', {}) if context else {}
            raw_nutrients = categories.get('nutrients', {})

            # Use calculations from previous agent or fallback to raw data
            weight = float(p_eval.get('weight') or patient.get('peso_v22') or 70.0)
            bmi = float(p_eval.get('bmi') or 25.0)
            
            # 2. Deterministic Targets
            protein_target = self.calculate_protein_needs(weight, bmi)
            fiber_target = 30.0 # Clinical baseline for microbiome health
            
            # 3. Gap Analysis
            # 'proteins_g' would be coming from the intake questionnaire in the 'nutrients' category
            actual_protein = self._clean_val(raw_nutrients.get('proteins_g', 0))
            protein_gap = round(protein_target - actual_protein, 1)

            # 4. LLM Synthesis (Clinical Rationale)
            prompt = (
                f"System: Clinical Dietitian Analyst.\n"
                f"Context: Patient Weight {weight}kg, BMI {bmi}.\n"
                f"Targets: {protein_target}g Protein, {fiber_target}g Fiber.\n"
                f"Findings: Current protein gap is {protein_gap}g.\n"
                f"Task: Provide a 1-sentence rationale explaining the protein target relative to their BMI.\n"
                f"Output: One clinical sentence only."
            )
            
            rationale = self._call_ollama(prompt)

            # 5. Build Result (Aligned for Stage 4 Planner)
            analysis_data = {
                "protein_target_g": protein_target,
                "fiber_target_g": fiber_target,
                "protein_gap_g": protein_gap,
                "actual_protein_g": actual_protein,
                "clinical_rationale": rationale,
                "weight": weight,
                "bmi": bmi
            }

            res = {
                "status": "success", 
                "summary": rationale, 
                "data": analysis_data,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self._save_to_local_memory(pid, res)
            return res

        except Exception as e:
            return {"status": "error", "message": f"NutrientsEvaluator failed: {str(e)}"}

    def calculate_protein_needs(self, weight: float, bmi: float) -> float:
        """Determines protein requirements using clinical BMI-adjusted multipliers."""
        # 1.2g/kg for Obesity to maintain LBM, 0.8g/kg for standard maintenance
        multiplier = 1.2 if bmi >= 30 else 0.8
        return round(float(weight) * multiplier, 1)

    def _check_local_memory(self, patient_id: str) -> Optional[Dict]:
        path = os.path.join(self.memory_dir, f"{patient_id}.json")
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

    def _save_to_local_memory(self, patient_id: str, data: Dict):
        path = os.path.join(self.memory_dir, f"{patient_id}.json")
        history = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try: history = json.load(f)
                except: history = []
        history.append(data)
        with open(path, 'w', encoding='utf-8') as f: json.dump(history[-10:], f, indent=4)

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt], 
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            raw = result.stdout.strip()
            return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        except: 
            return "Metabolic protein targets prioritized based on body mass index and clinical maintenance requirements."