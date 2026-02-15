# src/agents/microbiome_modulation_agent.py

import json
import os
import datetime
import subprocess
import re
from typing import Any, Dict, Optional

class MicrobiomeModulationAgent:
    def __init__(self, model_name: str = "gpt-oss:20b", memory_dir: str = "data/memory/microbiome"):
        self.name = "Microbiome Modulation"
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
            # 1. Extraction with clinical fallback logic
            micro_data = categories.get('microbiome', {})
            nutrients_data = categories.get('nutrients', {})

            def safe_float(val):
                if val is None or val == "": return 0.0
                try: return float(str(val).replace(',', '.'))
                except: return 0.0

            f = safe_float(micro_data.get('p_firmicutes', 0))
            b = safe_float(micro_data.get('p_bacteroidota', 0))
            
            # Calculate Ratio
            fb_ratio = round(f / b, 2) if b > 0 else 0.0
            status = self.classify_fb_ratio(fb_ratio)
            
            # Extract Dietary Fiber intake from raw data or previous evaluator
            fiber_val = safe_float(nutrients_data.get('dietary_fibre_g', 0))

            # 2. Sequential Handoff Integration (Stateful context)
            p_eval = context.get('personal_evaluator', {}).get('data', {}) if context else {}
            n_eval = context.get('nutrients_evaluator', {}).get('data', {}) if context else {}
            
            bmi = p_eval.get('bmi', 'N/A')
            fiber_target = n_eval.get('fiber_target_g', 30.0)

            # 3. LLM Clinical Interpretation
            prompt = (
                f"System: Microbiome Specialist.\n"
                f"Patient Status: BMI {bmi}. F/B Ratio: {fb_ratio} ({status}).\n"
                f"Current Fiber Intake: {fiber_val}g (Target: {fiber_target}g).\n"
                f"Task: Provide a 1-sentence rationale for gut health modulation targeting microbial diversity.\n"
                f"Output: Only the rationale sentence."
            )
            
            rationale = self._call_ollama(prompt)
            
            # Strategic Goal Selection for Stage 4 Planner
            mod_goal = "Optimize prebiotic variety and polyphenol intake"
            if fb_ratio > 2.0: 
                mod_goal = "Targeted reduction of Firmicutes via high-fiber and fermented foods"
            elif fb_ratio > 0 and fb_ratio < 0.5: 
                mod_goal = "Expansion of microbial diversity and Bacteroidetes support"
            elif fb_ratio == 0:
                mod_goal = "General microbial enrichment through diverse plant fiber variety"

            # 4. Construct Result
            analysis_data = {
                "fb_ratio": fb_ratio,
                "fb_status": status,
                "fiber_intake_g": fiber_val,
                "fiber_target_g": fiber_target,
                "microbial_rationale": rationale,
                "modulation_goal": mod_goal
            }

            res = {
                "status": "success", 
                "summary": rationale, 
                "data": analysis_data,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # 5. Update Agent Memory (Stateful Persistence)
            self._save_to_local_memory(pid, res)
            return res

        except Exception as e:
            return {"status": "error", "message": f"MicrobiomeAgent failed: {str(e)}"}

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
            return "Gut modulation should focus on prebiotic variety to restore microbial diversity."

    def classify_fb_ratio(self, ratio: float) -> str:
        if ratio <= 0: return 'Data Not Available'
        if ratio < 0.8: return 'Low (Bacteroidetes dominance)'
        elif ratio <= 1.2: return 'Balanced'
        elif ratio <= 2.0: return 'Moderate Firmicutes dominance'
        return 'High Firmicutes dominance (associated with adiposity)'