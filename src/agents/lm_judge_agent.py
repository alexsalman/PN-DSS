# src/agents/lm_judge_agent.py

import subprocess
import re
import datetime
import json
import os
from typing import Dict, Any

class LMJudgeAgent:
    """
    Stage 5: Clinical Critic (Second Opinion).
    Performs a strict audit comparing the generated meal plan against clinical targets.
    Forces a structured Markdown table output for the final report.
    """
    def __init__(self, model_name="gpt-oss:20b"):
        self.name = "LM Judge"
        self.role = "Critic/Auditor"
        self.model_name = model_name
        self.memory_dir = "memory"

    def run(self, context: Dict) -> Dict:
        pid = context.get('metadata', {}).get('patient_id', 'unknown')
        memory_path = f"{self.memory_dir}/{pid}.json"
        
        # 1. DATA HARVESTING (Blackboard Access)
        memories = {}
        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                full_memory = json.load(f)
                memories = full_memory.get("agent_memories", {})

        planner_data = memories.get('meal_planner_data', {})
        nutrient_data = memories.get('nutrients_evaluator_data', {})
        
        meal_plan = planner_data.get('meal_plan', '')
        
        # Priority: Use specific specialist targets from memory to ensure consistency
        prot = nutrient_data.get('protein_target_g', planner_data.get('clinical_targets', {}).get('protein', '70'))
        cal = nutrient_data.get('energy_target_kcal', planner_data.get('clinical_targets', {}).get('calories', '2000'))
        fib = nutrient_data.get('fiber_target_g', planner_data.get('clinical_targets', {}).get('fiber', '30'))

        print(f"\n[DEBUG_AGENT_IN: {self.name}] Performing Clinical Audit for PID: {pid}")

        try:
            if not meal_plan:
                return {"status": "error", "data": {"verdict": "ERROR", "critique": "Meal plan missing."}}

            # 2. THE AUDIT PROMPT
            # We explicitly instruct the model to perform a comparison and fill the table.
            prompt = (
                f"System: You are a Clinical Audit Bot. You only output structured text.\n"
                f"Task: Strictly compare the meal plan below against these 3 Clinical Targets:\n"
                f"1. Energy: {cal} kcal/day\n"
                f"2. Protein: {prot} g/day\n"
                f"3. Fiber: {fib} g/day\n\n"
                f"[MEAL PLAN]\n{meal_plan[:3000]}\n\n"
                f"REQUIRED RESPONSE FORMAT (NO OTHER TEXT):\n"
                f"VERDICT: [Safe, Caution, or Unsafe]\n"
                f"CRITIQUE: [One sentence clinical reason for the verdict]\n"
                f"TABLE:\n"
                f"| Metric | Target | Actual (avg/day) | Status | Impact |\n"
                f"| Calories | {cal} | [Val] | [P/F] | [Impact] |\n"
                f"| Protein | {prot} | [Val] | [P/F] | [Impact] |\n"
                f"| Fiber | {fib} | [Val] | [P/F] | [Impact] |"
            )

            raw_response = self._call_ollama(prompt)

            # 3. EXTRACTION & VALIDATION
            # Verdict Extraction with Whitelist Filter
            verdict_match = re.search(r"VERDICT:\s*(\w+)", raw_response, re.I)
            raw_verdict = verdict_match.group(1).strip().upper() if verdict_match else "CAUTION"
            
            # Prevent "Verdict: CALORIES" error:
            if raw_verdict not in ["SAFE", "CAUTION", "UNSAFE"]:
                verdict = "UNSAFE" # Conservative fallback for safety
            else:
                verdict = raw_verdict

            # Critique Extraction (Stripping surrounding conversational fluff)
            critique_match = re.search(r"CRITIQUE:(.*?)(?=TABLE|$)", raw_response, re.S | re.I)
            critique = critique_match.group(1).strip() if critique_match else "Clinical audit completed."
            
            # Clean critique of markdown or internal thinking if it leaked
            critique = re.sub(r'[*_#]', '', critique)

            # Table Extraction
            table_start = raw_response.find('|')
            structured_table = raw_response[table_start:].strip() if table_start != -1 else "No audit table."

            audit_results = {
                "verdict": verdict,
                "critique": critique,
                "structured_comparison": structured_table,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self._update_memory(memory_path, audit_results)

            return {"status": "success", "data": audit_results}

        except Exception as e:
            return {"status": "error", "message": f"Audit Engine Failed: {str(e)}"}

    def _update_memory(self, path: str, audit_data: Dict):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                memory_json = json.load(f)
            memory_json.setdefault("agent_memories", {})["lm_judge_data"] = audit_data
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(memory_json, f, indent=4)

    def _call_ollama(self, prompt: str) -> str:
        try:
            # Setting environment variables to force consistent encoding
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True, 
                text=True, 
                check=True, 
                encoding='utf-8',
                env=env
            )
            
            # Clean all thinking artifacts immediately
            output = result.stdout
            output = re.sub(r'<(think|thought)>.*?</\1>', '', output, flags=re.DOTALL | re.IGNORECASE)
            output = re.sub(r'(?i)Thinking\.\.\..*?(\.\.\.done thinking|Let\'s think).*?\.?', '', output, flags=re.DOTALL)
            
            return output.strip()
        except Exception as e:
            print(f"[ERROR] Ollama subprocess failed: {e}")
            return "VERDICT: UNSAFE\nCRITIQUE: Audit system interface error.\nTABLE: | Error | Audit Failed |"
