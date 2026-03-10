# src/agents/meal_planner_agent.py

import json
import os
import subprocess
import re
from typing import Dict, List, Any, Optional

class MealPlannerAgent:
    """
    Planning Stage (Stage 4):
    Translates clinical targets into a specific dietary schedule.
    Acts as the creative synthesizer of the specialists' recommendations.
    """
    def __init__(self, 
                 model_name="gpt-oss:20b",
                 drv_path: str = 'data/DRVs_Adults.json',
                 sources_path: str = 'data/Dietary_Sources.json'):
        
        self.name = "Meal Planner"
        self.role = "Planning"
        self.model_name = model_name
        self.memory_dir = "memory" 
        self.drvs = self._load_structured_data(drv_path)
        self.food_db = self._load_structured_data(sources_path)

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        memory_path = f"{self.memory_dir}/{pid}.json"

        try:
            # 1. HARVEST: Access shared memory (The Blackboard)
            agent_memories = {}
            if os.path.exists(memory_path):
                with open(memory_path, 'r', encoding='utf-8') as f:
                    full_memory = json.load(f)
                    agent_memories = full_memory.get("agent_memories", {})

            # Pulling finalized data from previous specialists
            p_data = agent_memories.get('personal_evaluator_data', {})
            n_data = agent_memories.get('nutrients_evaluator_data', {})
            s_data = agent_memories.get('constraint_manager_data', {})
            
            user_query = context.get('user_query') or "Generate a nutritional plan."

            # DURATION EXTRACTION (e.g., "7-day plan")
            days_match = re.search(r'(\d+)\s*-?day', user_query.lower())
            plan_days = int(days_match.group(1)) if days_match else 7

            # CLINICAL TARGET RECOVERY - NO DEFAULTS to prevent audit mismatch
            # We use the specific calculated targets from the Nutrients/Personal specialists
            bmi_val = p_data.get('bmi', "N/A")
            kcal_val = n_data.get('energy_target_kcal') or p_data.get('tee_kcal') or "2000"
            prot_val = n_data.get('protein_target_g', "70")
            fib_val = n_data.get('fiber_target_g', "30")

            # SAFETY FILTERING: Build guide of allowed foods based on specialist exclusions
            exclusions = f"{s_data.get('strict_exclusions', '')}".lower()
            safe_categories = self._get_safe_food_list(exclusions)
            source_guide = "\n".join([f"- {s['nutrient']} ({s['category']}): {s['food_sources']}" for s in safe_categories[:12]])

            # 2. HIGH-PRECISION PROMPT (Updated for Caloric Compliance)
            prompt = (
                f"System: You are a Clinical Nutritionist. You ONLY output Markdown tables.\n"
                f"User Request: {user_query}\n\n"
                f"[CLINICAL TARGETS - MANDATORY]\n"
                f"- Daily Energy: {kcal_val} kcal (Must NOT fall below this)\n"
                f"- Daily Protein: {prot_val}g\n"
                f"- Daily Fiber: {fib_val}g\n"
                f"- Patient BMI: {bmi_val}\n\n"
                f"[VALIDATED FOOD SOURCES]\n{source_guide}\n\n"
                f"TASK: Generate a {plan_days}-day meal plan that meets the caloric target of {kcal_val} kcal/day.\n"
                f"REQUIRED FORMAT: A single Markdown table.\n"
                f"COLUMNS: | Day | Breakfast | Lunch | Dinner |\n"
                f"RULES: \n"
                f"1. Start immediately with the table header.\n"
                f"2. Use <br> for line breaks within cells.\n"
                f"3. NO introductory text, NO self-instructions, NO closing remarks.\n"
                f"4. Ensure the total energy density matches {kcal_val} kcal exactly."
            )

            # 3. LLM Execution & Secondary Cleaning
            raw_meal_plan = self._call_ollama(prompt)
            
            # Remove any markdown code block wrappers
            clean_plan = re.sub(r'```markdown|```', '', raw_meal_plan).strip()
            
            # Ensure the table actually starts at the first pipe
            if '|' in clean_plan:
                clean_plan = clean_plan[clean_plan.find('|'):]

            planning_results = {
                "meal_plan": clean_plan,
                "clinical_targets": {
                    "calories": kcal_val,
                    "protein": prot_val,
                    "fiber": fib_val,
                    "bmi": bmi_val 
                },
                "duration_days": plan_days
            }

            # 4. BLACKBOARD WRITE
            self._update_memory(memory_path, planning_results, plan_days)

            return {"status": "success", "data": planning_results}

        except Exception as e:
            return {"status": "error", "message": f"Planning failed: {str(e)}"}

    def _update_memory(self, path: str, plan_data: Dict, days: int):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                memory_json = json.load(f)
            
            if "agent_memories" not in memory_json:
                memory_json["agent_memories"] = {}
                
            # Log rationale for clinical trace
            memory_json["agent_memories"]["meal_planner_memory"] = f"{days}-day plan generated."
            memory_json["agent_memories"]["meal_planner_data"] = plan_data
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(memory_json, f, indent=4)

    def _get_safe_food_list(self, exclusion_str: str) -> List[Dict]:
        """Filters the food database against patient allergies/constraints."""
        if not self.food_db: return []
        exc_list = [e.strip().lower() for e in re.split(r'[,\s]+', exclusion_str) if len(e.strip()) > 2]
        safe = []
        for entry in self.food_db:
            sources_text = entry.get('food_sources', '').lower()
            if not any(exc in sources_text for exc in exc_list):
                safe.append(entry)
        return safe

    def _load_structured_data(self, path: str) -> List[Dict]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return []
        return []

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(["ollama", "run", self.model_name, prompt],
                                 capture_output=True, text=True, check=True, encoding='utf-8')
            
            output = result.stdout.strip()
            # AGGRESSIVE CLEANING of thinking blocks/leaks
            output = re.sub(r'<(think|thought)>.*?</\1>', '', output, flags=re.DOTALL | re.IGNORECASE)
            output = re.sub(r'(?i)Thinking\.\.\..*?(\.\.\.done thinking|Let\'s think).*?\.?', '', output, flags=re.DOTALL)
            
            return output.strip()
        except Exception:
            return "Plan generation error. Interface with LLM failed."