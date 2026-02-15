# src/agents/meal_planner_agent.py

import json
import os
import datetime
import subprocess
import re
from typing import Dict, List, Any, Optional

class MealPlannerAgent:
    """
    Stage 4 Planning Agent (Stateful):
    Synthesizes a 7-day clinical plan by matching Reasoning Phase outcomes 
    with deterministic Dietary Reference Values (DRVs) and Food Source databases.
    """
    def __init__(self, 
                 model_name="gpt-oss:20b",
                 drv_path: str = 'data/DRVs_Adults.json',
                 sources_path: str = 'data/Dietary_Sources.json',
                 memory_dir: str = "data/memory/planner"):
        
        self.name = "Meal Planner"
        self.role = "Planning"
        self.model_name = model_name
        self.memory_dir = memory_dir
        
        os.makedirs(self.memory_dir, exist_ok=True)

        # Initialize structured clinical lookup tools
        self.drvs = self._load_structured_data(drv_path)
        self.food_db = self._load_structured_data(sources_path)

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        print(f"\n[DEBUG_AGENT_IN: {self.name}] PID: {pid}")

        # --- AGENT MEMORY OPTIMIZATION (STATEFUL) ---
        cached_data = self._check_local_memory(pid)
        if cached_data:
            print(f"[MEMORY_HIT: {self.name}] Using existing plan for PID: {pid}")
            return cached_data

        try:
            # 1. State Retrieval from Reasoning Stages
            p_eval = context.get('personal_evaluator', {}).get('data', {})
            n_eval = context.get('nutrients_evaluator', {}).get('data', {})
            m_eval = context.get('microbiome_modulation', {}).get('data', {})
            safety = context.get('constraint_manager', {}).get('data', {})
            
            user_query = context.get('user_query') or "Generate a nutritional plan."

            # 2. Deterministic Tool: DRV Target Matching
            age = float(str(p_eval.get('age') or patient.get('edad_v22') or 30))
            gender_val = str(p_eval.get('sex') or patient.get('sexo', '2'))
            gender = "Male" if gender_val == '1' else "Female"
            
            clinical_targets = self._get_matched_drvs(age, gender)

            # 3. Deterministic Tool: Dietary Source Filtering
            # Merging strict exclusions and general allergies for the search tool
            exclusions = f"{safety.get('strict_exclusions', '')} {safety.get('allergies', '')}".lower()
            safe_categories = self._get_safe_food_list(exclusions)
            
            source_guide = "\n".join([f"- {s['nutrient']} ({s['category']}): {s['food_sources']}" for s in safe_categories[:15]])

            # 4. Construct Context-Aware Prompt
            prompt = (
                f"System: Clinical Nutritionist synthesis.\n"
                f"User Request: {user_query}\n\n"
                f"[CLINICAL MACRO TARGETS]\n"
                f"- Energy: {p_eval.get('tee_kcal', 2000)} kcal\n"
                f"- Protein: {n_eval.get('protein_target_g', 60)}g\n"
                f"- Fiber: {n_eval.get('fiber_target_g', 30)}g\n\n"
                f"[CLINICAL MICRO TARGETS (DRVs for {gender}, Age {age})]\n"
                f"{clinical_targets}\n\n"
                f"[VALIDATED SAFE FOOD SOURCES]\n"
                f"{source_guide}\n\n"
                f"[SAFETY CONSTRAINTS & RATIONALE]\n"
                f"- Constraints: {safety.get('clinical_safety_summary', 'Standard precautions')}\n"
                f"- Metabolic Goal: {p_eval.get('weight_trend_rationale', '')}\n"
                f"- Microbiome Goal: {m_eval.get('modulation_goal', '')}\n\n"
                f"Requirements: Output a 7-day meal plan in Markdown table format. Include specific breakfast, lunch, and dinner suggestions using the validated food sources. No conversational filler."
            )

            # 5. LLM Execution
            meal_plan = self._call_ollama(prompt)

            # 6. Result Construction (Ready for Stage 5 Judge and Stage 6 Report)
            res = {
                "status": "success",
                "summary": f"Plan synthesized for {gender} ({age}y) targeting {p_eval.get('tee_kcal')} kcal.",
                "data": {
                    "meal_plan": meal_plan,
                    "clinical_targets": {
                        "calories": p_eval.get('tee_kcal', 2000),
                        "protein": n_eval.get('protein_target_g', 60),
                        "fiber": n_eval.get('fiber_target_g', 30),
                        "micro_drvs": clinical_targets
                    },
                    "agent_summaries": {
                        "personal": p_eval.get('weight_trend_rationale', 'N/A'),
                        "nutrients": n_eval.get('clinical_rationale', 'N/A'),
                        "microbiome": m_eval.get('microbial_rationale', 'N/A'),
                        "constraints": safety.get('clinical_safety_summary', 'N/A')
                    }
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self._save_to_local_memory(pid, res)
            return res

        except Exception as e:
            return {"status": "error", "message": f"Planning failed: {str(e)}"}

    def _get_safe_food_list(self, exclusion_str: str) -> List[Dict]:
        """Filters Dietary_Sources.json by excluding allergen-containing rows."""
        if not self.food_db: return []
        # Split by comma or space and remove empty strings
        exc_list = [e.strip().lower() for e in re.split(r'[,\s]+', exclusion_str) if len(e.strip()) > 2]
        
        safe = []
        for entry in self.food_db:
            sources_text = entry.get('food_sources', '').lower()
            category_text = entry.get('category', '').lower()
            # If no exclusion word is found in the sources or category name
            if not any(exc in sources_text or exc in category_text for exc in exc_list):
                safe.append(entry)
        return safe

    def _get_matched_drvs(self, age: float, gender: str) -> str:
        """Finds matching micronutrient targets in DRVs_Adults.json."""
        matches = []
        for row in self.drvs:
            # Match Gender and check Age Range (e.g., '>= 18 years')
            if str(row.get('gender')).capitalize() == gender.capitalize():
                age_req = row.get('age', '')
                if ">=" in age_req and age >= 18:
                    # Prefer PRI (Population Reference Intake) over AR (Average Requirement)
                    pri = str(row.get('population_reference_intake_(pri)', 'NA'))
                    ar = str(row.get('average_requirement_(ar)', 'NA'))
                    val = pri if "NA" not in pri else ar
                    if "NA" not in val:
                        matches.append(f"- {row['nutrient']}: {val}")
        return "\n".join(matches[:12]) if matches else "Standard Adult Micronutrient Guidelines"

    def _load_structured_data(self, path: str) -> List[Dict]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return []
        return []

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
        with open(path, 'w', encoding='utf-8') as f: json.dump(history[-5:], f, indent=4)

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            raw = result.stdout.strip()
            return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        except:
            return "Plan generation error. Please verify terminal connectivity to Ollama."