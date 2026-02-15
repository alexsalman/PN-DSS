# src/agents/lm_judge_agent.py

import subprocess
import re
import datetime
from typing import Dict, Any

class LMJudgeAgent:
    """
    Stage 5: Clinical Critic (Second Opinion).
    Stateless: Performs a unidirectional audit of the Plan vs. the reasoning context.
    Ensures metabolic goals, fiber targets, and safety exclusions are met without 
    storing local history.
    """
    def __init__(self, model_name="gpt-oss:20b"):
        self.name = "LM Judge"
        self.role = "Critic/Auditor"
        self.model_name = model_name

    def run(self, context: Dict) -> Dict:
        # 1. State Extraction (Volatile context only)
        planner_entry = context.get('meal_planner', {})
        planner_data = planner_entry.get('data', {})
        meal_plan = planner_data.get('meal_plan', '')
        
        # Pull clinical targets and safety constraints
        targets = planner_data.get('clinical_targets', {})
        safety_context = context.get('constraint_manager', {}).get('data', {})
        
        pid = context.get('metadata', {}).get('patient_id', 'unknown')

        print(f"\n[DEBUG_AGENT_IN: {self.name}] Auditing Final Plan for PID: {pid}")

        try:
            if not meal_plan or len(meal_plan) < 50:
                return {
                    "status": "error",
                    "data": {
                        "verdict": "SKIPPED", 
                        "critique": "No meal plan generated to audit."
                    }
                }

            # 2. Audit Prompt Engineering
            prot = targets.get('protein', 'N/A')
            cal = targets.get('calories', 'N/A')
            fib = targets.get('fiber', 'N/A')
            excl = safety_context.get('allergies', 'None reported')

            prompt = (
                f"System: You are an Independent Clinical Auditor.\n"
                f"Task: Audit the proposed meal plan against clinical requirements.\n\n"
                f"[CLINICAL TARGETS]\n"
                f"- Calories: {cal} kcal\n"
                f"- Protein: {prot}g\n"
                f"- Fiber: {fib}g\n"
                f"- Mandatory Exclusions: {excl}\n\n"
                f"[PROPOSED 7-DAY PLAN]\n"
                f"{meal_plan[:3500]}\n\n"
                f"[AUDIT INSTRUCTIONS]\n"
                f"1. SAFETY: Does any meal contain an excluded allergen?\n"
                f"2. NUTRITIONAL ALIGNMENT: Are the calorie/protein/fiber targets realistic?\n"
                f"3. VERDICT: Choose one: 'Safe', 'Caution', or 'Unsafe'.\n"
                f"4. CRITIQUE: Provide 2-3 specific clinical observations.\n\n"
                f"Response Format: Start with 'VERDICT: [Safe/Caution/Unsafe]' followed by your critique."
            )

            print(f"[DEBUG_AGENT_LLM_PROMPT: {self.name}] Synthesizing second opinion...")
            critique_response = self._call_ollama(prompt)

            # 3. Enhanced Verdict Parsing Logic
            verdict = "Caution" 
            upper_response = critique_response.upper()
            
            if "VERDICT: SAFE" in upper_response:
                verdict = "Safe"
            elif "VERDICT: UNSAFE" in upper_response:
                verdict = "Unsafe"
            elif "VERDICT: CAUTION" in upper_response:
                verdict = "Caution"

            # 4. Construct Result Payload (No saving to local memory)
            audit_data = {
                "verdict": verdict,
                "critique": critique_response,
                "timestamp": datetime.datetime.now().isoformat()
            }

            return {
                "status": "success",
                "summary": f"Audit complete. Verdict: {verdict}",
                "data": audit_data
            }

        except Exception as e:
            print(f"[DEBUG_AGENT_ERROR: {self.name}] {str(e)}")
            return {
                "status": "error",
                "message": f"Judge Agent failed: {str(e)}",
                "data": {"verdict": "Error", "critique": f"System error during audit: {str(e)}"}
            }

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            raw = result.stdout.strip()
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            return cleaned.strip()
        except:
            return "VERDICT: Caution. Critique: The automated audit could not be performed due to a system timeout."