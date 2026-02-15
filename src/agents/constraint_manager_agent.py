# src/agents/constraint_manager_agent.py

import json
import os
import datetime
import subprocess
import re
from typing import Dict, List, Any, Optional

class ConstraintManagerAgent:
    def __init__(self, model_name: str = "gpt-oss:20b", memory_dir: str = "data/memory/constraints"):
        self.name = "Constraint Manager"
        self.role = "Reasoning"
        self.model_name = model_name
        self.memory_dir = memory_dir
        
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Clinical reference rules for LLM synthesis
        self.condition_rules = {
            "Diabetes": "Low Glycemic Index; limit simple sugars; stable carb distribution.",
            "Hypertension": "Sodium < 2300mg/day; high potassium; DASH principles.",
            "Hypercholesterolemia": "Saturated fats < 7% total kcal; increase soluble fiber.",
            "IBD/Crohn's": "Low-residue during active flares; focus on hydration.",
            "Obesity": "Caloric deficit; high satiety; portion control."
        }

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        print(f"\n[DEBUG_AGENT_IN: {self.name}] PID: {pid}")

        # --- AGENT MEMORY OPTIMIZATION (STATEFUL) ---
        cached_data = self._check_local_memory(pid)
        if cached_data:
            print(f"[MEMORY_HIT: {self.name}] Using existing safety analysis for PID: {pid}")
            return cached_data

        try:
            # 1. Extraction of Safety Data
            constraints_data = categories.get('constraints', {})
            active_conditions = []
            active_meds = []
            
            # Expanded mapping for clinical robustness
            field_labels = {
                "met_diab_v22": "Diabetes",
                "hat_med_v22": "Hypertension",
                "col_med_v22": "High Cholesterol",
                "cron_med_v22": "Crohn's Disease",
                "diabetico_v22": "Diabetic Status"
            }

            for key, val in constraints_data.items():
                if val and str(val).lower() not in ['0', 'no', 'none', '', 'nan']:
                    label = field_labels.get(key, key)
                    if "med" in key.lower():
                        active_meds.append(f"{label}: {val}")
                    else:
                        active_conditions.append(label)

            pathologies_str = ", ".join(active_conditions) if active_conditions else "None reported"
            meds_str = ", ".join(active_meds) if active_meds else "None reported"
            
            # Allergy extraction (Crucial for Stage 4 deterministic filtering)
            raw_allergies = str(patient.get('allergies', 'None reported')).lower()
            clean_allergies = "None" if raw_allergies in ['0', 'no', 'none', ''] else raw_allergies

            # 2. Historical Safety Reconciliation
            history_data = context.get('history', {}) if context else {}
            past_visits = history_data.get('visit_history', [])
            historical_alerts = list(set([str(v.get('verdict')) for v in past_visits if v.get('verdict') == "Review"]))
            history_context = f"Prior Safety Flags: {'; '.join(historical_alerts)}" if historical_alerts else "No prior alerts."

            # 3. LLM Safety Synthesis
            prompt = (
                f"System: Clinical Safety Auditor.\n"
                f"Conditions: {pathologies_str}. Medications: {meds_str}. Allergies: {clean_allergies}.\n"
                f"{history_context}\n"
                f"Reference Rules: {json.dumps(self.condition_rules)}\n"
                f"Task: Synthesize 1-2 sentences of mandatory dietary constraints for the meal planner.\n"
                f"Output: Only the constraints. No conversational filler."
            )
            
            safety_summary = self._call_ollama(prompt)

            # 4. Construct Result (Isolated keys for Stage 4 Planner)
            safety_data = {
                "active_pathologies": active_conditions,
                "active_medications": active_meds,
                "allergies": clean_allergies,
                "clinical_safety_summary": safety_summary,
                "strict_exclusions": clean_allergies # Explicit key for Meal Planner search tool
            }

            res = {
                "status": "success", 
                "summary": safety_summary, 
                "data": safety_data,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # 5. Update Agent Memory (Stateful Persistence)
            self._save_to_local_memory(pid, res)
            return res

        except Exception as e:
            return {"status": "error", "message": f"ConstraintManager failed: {str(e)}"}

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
            return "Prioritize DASH principles and allergen avoidance."