# src/agents/constraint_manager_agent.py

import json
import os
import subprocess
import re
from typing import Dict, List, Any, Optional

class ConstraintManagerAgent:
    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.name = "Constraint Manager"
        self.role = "Reasoning"
        self.model_name = model_name
        self.memory_dir = "memory" 
        
        # RESTORED: Full Clinical Mapping Rules
        self.mapping_rules = {
            "Obesity": "Energy-restricted diet (500kcal deficit); Carbs 50-55%, Proteins 15-20%, Fats 30-35%; Low energy density foods; reduce sweets/refined sugars.",
            "Elevated Waist-Hip": "Moderate trans/saturated fats; reduce sugars and sweets.",
            "Hypertension": "Sodium-restricted (1000-2000mg/day); avoid processed foods; steaming/stewing cooking techniques.",
            "High Cholesterol": "Increase soluble fiber (oats, beans); omega-3 fats (salmon, tuna); plant-based proteins; whole grains.",
            "Hyperuricemia": "Low-purine diet; limit red meat/organ meats; avoid alcohol/sugary beverages.",
            "Drug-Nutrient": "Statin intake: Avoid grapefruit juice and Vitamin B3 supplementation.",
            "Low F/B Ratio": "Incorporate Prebiotics: Inulin, Short-chain Oligosaccharides (chicory, garlic, leeks, asparagus), and Curcumin."
        }

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        pid = str(patient.get('record_id', 'unknown'))
        memory_path = f"{self.memory_dir}/{pid}.json"

        context = context or {}
        bmi = context.get('bmi', 'Unknown')
        fb_ratio = context.get('fb_ratio', 0.0)

        try:
            # 1. Extraction of Safety Data
            constraints_data = categories.get('constraints', {})
            active_conditions = []
            active_meds = []
            applied_clinical_rules = []
            
            field_labels = {
                "met_diab_v22": "Diabetes",
                "hat_med_v22": "Hypertension",
                "col_med_v22": "High Cholesterol",
                "diabetico_v22": "Diabetic Status",
                "statin_use": "Statin Intake"
            }

            for key, val in constraints_data.items():
                if val and str(val).lower() not in ['0', 'no', 'none', '', 'nan']:
                    label = field_labels.get(key, key)
                    if "med" in key.lower() or "statin" in key.lower(): 
                        active_meds.append(label)
                    else: 
                        active_conditions.append(label)

            # 2. Apply FULL Clinical Mapping Rules 
            if isinstance(bmi, (int, float)) and bmi >= 30:
                applied_clinical_rules.append(self.mapping_rules["Obesity"])
            
            if "Hypertension" in active_conditions:
                applied_clinical_rules.append(self.mapping_rules["Hypertension"])
                
            if "High Cholesterol" in active_conditions:
                applied_clinical_rules.append(self.mapping_rules["High Cholesterol"])

            if "Statin Intake" in active_meds:
                applied_clinical_rules.append(self.mapping_rules["Drug-Nutrient"])

            if 0 < fb_ratio < 1.57: 
                applied_clinical_rules.append(self.mapping_rules["Low F/B Ratio"])

            # 3. Allergy Extraction
            raw_allergies = str(patient.get('allergies', 'None reported')).lower()
            clean_allergies = "None" if raw_allergies in ['0', 'no', 'none', ''] else raw_allergies

            # 4. LLM Safety Synthesis
            rules_context = " | ".join(applied_clinical_rules)
            prompt = (
                f"System: Clinical Safety Auditor. Output ONLY 1-2 sentences of mandatory constraints.\n"
                f"Conditions: {', '.join(active_conditions)}. Meds: {', '.join(active_meds)}.\n"
                f"Allergies: {clean_allergies}. Clinical Rules: {rules_context}.\n"
                f"Output:"
            )
            
            new_safety_summary = self._call_ollama(prompt)

            # 5. Build Result Data
            safety_data = {
                "active_pathologies": active_conditions,
                "active_medications": active_meds,
                "applied_rules": applied_clinical_rules,
                "clinical_safety_summary": new_safety_summary,
                "strict_exclusions": clean_allergies 
            }

            # 6. BLACKBOARD WRITE
            self._update_memory(memory_path, new_safety_summary, safety_data)

            return {
                "status": "success", 
                "summary": new_safety_summary, 
                "data": safety_data
            }

        except Exception as e:
            return {"status": "error", "message": f"ConstraintManager failed: {str(e)}"}

    def _update_memory(self, path: str, summary: str, data: Dict):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                memory_json = json.load(f)
            if "agent_memories" not in memory_json:
                memory_json["agent_memories"] = {}
            memory_json["agent_memories"]["constraint_manager_memory"] = summary
            memory_json["agent_memories"]["constraint_manager_data"] = data
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(memory_json, f, indent=4)

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(["ollama", "run", self.model_name, prompt], 
                                 capture_output=True, text=True, check=True, encoding='utf-8')
            clean_text = re.sub(r'<(think|thought)>.*?</\1>', '', result.stdout.strip(), flags=re.DOTALL | re.IGNORECASE).strip()
            lines = [l for l in clean_text.split('\n') if l.strip()]
            return lines[-1] if lines else "Adhere to strict sodium limits and allergen avoidance."
        except: 
            return "Adhere to strict sodium limits and allergen avoidance."