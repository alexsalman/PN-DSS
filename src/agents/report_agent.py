# src/agents/report_agent.py

import datetime
import subprocess
import re
from typing import Dict, Any

class ReportAgent:
    """
    Response Stage (Stage 6):
    Final Clinical Documenter.
    Stateless: Synthesizes the finalized, audited data into a structured 
    clinical report payload for the PDF Generator without storing history.
    """
    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.name = "Report Agent"
        self.role = "Response"
        self.model_name = model_name

    def run(self, patient: Dict, context: Dict = None) -> Dict:
        """
        Processes the consolidated context from the Supervisor into a 
        final report payload.
        """
        pid = str(patient.get('record_id', 'unknown'))
        print(f"\n[DEBUG_AGENT_IN: {self.name}] Synthesizing Final Clinical Report for PID: {pid}...")

        try:
            # 1. RECEIVE AUDITED PACKAGE
            planner_results = context.get('meal_planner', {}).get('data', {})
            judge_results = context.get('lm_judge', {}).get('data', {})
            
            # Extract reasoning summaries (Rationale)
            summaries = planner_results.get('agent_summaries', {})
            meal_plan = planner_results.get('meal_plan', 'Meal plan not generated.')
            
            # Independent safety verdict
            verdict = judge_results.get('verdict', 'N/A')
            critique = judge_results.get('critique', 'Audit feedback pending.')

            # 2. CLINICAL NARRATIVE SYNTHESIS
            prompt = (
                f"System: Senior Clinical Nutritionist Writing Lead.\n"
                f"Patient Profile: {pid}\n\n"
                f"[REASONING LOGIC]\n"
                f"- Personal/Metabolic: {summaries.get('personal', 'Stable weight maintenance.')}\n"
                f"- Nutrients: {summaries.get('nutrients', 'Standard macro distribution.')}\n"
                f"- Microbiome: {summaries.get('microbiome', 'General enrichment strategy.')}\n"
                f"- Clinical Rules: {summaries.get('constraints', 'No specific contraindications.')}\n\n"
                f"[AUDITOR SECOND OPINION]\n"
                f"Verdict: {verdict}\n"
                f"Audit Notes: {critique}\n\n"
                f"Task: Write a two-paragraph 'Professional Executive Summary'.\n"
                f"Paragraph 1: Summarize the nutritional strategy based on metabolic and microbiome needs.\n"
                f"Paragraph 2: Directly address the auditor's {verdict} verdict and cautionary notes.\n"
                f"Requirements: Professional medical tone. Markdown format. No conversational headers."
            )

            print(f"[DEBUG_AGENT_LLM_PROMPT: {self.name}] Generating professional narrative...")
            clinical_summary = self._call_ollama(prompt)

            # 3. CONSTRUCT FINAL PAYLOAD (PDF-Ready)
            report_payload = {
                "metadata": {
                    "patient_id": pid,
                    "report_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "status": verdict,
                    "age": patient.get('edad_v22', 'N/A'),
                    "bmi": context.get('personal_evaluator', {}).get('data', {}).get('bmi', 'N/A'),
                    "protein_target": planner_results.get('clinical_targets', {}).get('protein', 'N/A'),
                    "kcal_target": planner_results.get('clinical_targets', {}).get('calories', 'N/A')
                },
                "clinical_summary": clinical_summary,
                "judge_critique": critique,
                "meal_plan_table": meal_plan,
                "clinical_trace": context.get('agent_trace', []) 
            }

            # NOTE: Persistence logic (_save_to_local_memory) removed to maintain statelessness.

            return {
                "status": "success",
                "summary": f"Clinical document finalized with '{verdict}' verdict.",
                "data": report_payload
            }

        except Exception as e:
            print(f"[DEBUG_AGENT_ERROR: {self.name}] {str(e)}")
            return {
                "status": "error", 
                "message": f"ReportAgent failed: {str(e)}", 
                "data": {}
            }

    def _call_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            raw = result.stdout.strip()
            return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        except:
            return "Professional clinical summary currently unavailable."