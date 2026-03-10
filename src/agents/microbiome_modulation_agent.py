# src/agents/microbiome_modulation_agent.py

import json
import os
import datetime
import subprocess
import re
from typing import Any, Dict, Optional

class MicrobiomeModulationAgent:
    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.name = "Microbiome Modulation"
        self.role = "Reasoning"
        self.model_name = model_name
        self.memory_dir = "memory"

    def run(self, patient: Dict, categories: Dict, missing: Dict, context: Dict = None) -> Dict:
        """
        Executes microbiome analysis. 
        Context should contain 'bmi' and 'fiber_target' from previous agents.
        """
        pid = str(patient.get('record_id', 'unknown'))
        memory_path = f"{self.memory_dir}/{pid}.json"
        
        # Use context passed by the Supervisor to maintain the 'Blackboard' flow
        # instead of the agent performing its own file reads.
        context = context or {}
        bmi = context.get('bmi', 'N/A')
        fiber_target = context.get('fiber_target', 30.0)

        try:
            # 1. Extraction from Raw Data
            micro_data = categories.get('microbiome', {})
            
            def safe_float(val):
                if val is None or val == "": return 0.0
                try: return float(str(val).replace(',', '.'))
                except: return 0.0

            f_perc = safe_float(micro_data.get('p_firmicutes', 0))
            b_perc = safe_float(micro_data.get('p_bacteroidota', 0))
            
            # Calculate F/B Ratio
            fb_ratio = round(f_perc / b_perc, 2) if b_perc > 0 else 0.0
            
            # 2. Classification Logic (Strict Thresholds)
            status = self.classify_fb_ratio(fb_ratio)
            
            # Extract raw fiber intake for gap analysis
            fiber_val = safe_float(categories.get('nutrients', {}).get('dietary_fibre_g', 0))

            # 3. LLM Clinical Interpretation
            # We explicitly instruct the LLM to output ONLY the rationale to avoid 
            # the "Thinking..." leak seen in previous reports.
            prompt = (
                f"System: Microbiome Specialist. Output exactly one clinical sentence. No filler.\n"
                f"Context: Patient BMI is {bmi}. Target Fiber is {fiber_target}g.\n"
                f"Status: F/B Ratio is {fb_ratio} ({status}). Current Fiber: {fiber_val}g.\n"
                f"Task: Provide a 1-sentence clinical rationale for gut health modulation.\n"
                f"Output:"
            )
            
            new_rationale = self._call_ollama(prompt)
            
            # 4. Construct Result Data
            analysis_data = {
                "fb_ratio": fb_ratio,
                "fb_status": status,
                "fiber_intake_g": fiber_val,
                "fiber_target_g": fiber_target,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # 5. BLACKBOARD WRITE: Persist back to central JSON
            self._update_memory(memory_path, new_rationale, analysis_data)

            return {
                "status": "success", 
                "summary": new_rationale, 
                "data": analysis_data
            }

        except Exception as e:
            return {"status": "error", "message": f"MicrobiomeAgent failed: {str(e)}"}

    def classify_fb_ratio(self, ratio: float) -> str:
        """Exact classification logic for Firmicutes/Bacteroidetes ratio."""
        if ratio <= 0: 
            return 'Data Not Available'
        
        if ratio < 1.57:
            return 'lower ratio'
        elif 1.57 <= ratio <= 1.99:
            return 'normal ranges'
        else: # ratio > 1.99
            return 'high ratio'

    def _update_memory(self, path: str, summary: str, data: Dict):
        """Standardized Blackboard write operation."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                memory_json = json.load(f)
            
            if "agent_memories" not in memory_json:
                memory_json["agent_memories"] = {}
            
            memory_json["agent_memories"]["microbiome_modulation_memory"] = summary
            memory_json["agent_memories"]["microbiome_modulation_data"] = data
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(memory_json, f, indent=4)

    def _call_ollama(self, prompt: str) -> str:
        """Calls the LLM and aggressively strips any internal reasoning/filler."""
        try:
            result = subprocess.run(["ollama", "run", self.model_name, prompt],
                                 capture_output=True, text=True, check=True, encoding='utf-8')
            # Strip <think> tags and ensure only the last line (the rationale) is kept
            clean_text = re.sub(r'<think>.*?</think>', '', result.stdout.strip(), flags=re.DOTALL).strip()
            lines = [l for l in clean_text.split('\n') if l.strip()]
            return lines[-1] if lines else "Optimize fiber variety to support a healthy F/B ratio."
        except:
            return "Microbiome modulation should focus on prebiotic variety to restore microbial diversity."