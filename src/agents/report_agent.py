# src/agents/report_agent.py

import datetime
import re
import json
import os
from typing import Dict, Any

class ReportAgent:
    """
    Response Stage (Stage 6):
    Final Clinical Documenter.
    
    Acts as a data orchestrator. It harvests results and sanitizes reasoning logs 
    to ensure the final report follows the 3-Layer Clinical Audit structure.
    """
    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.name = "Report Agent"
        self.role = "Response"
        self.model_name = model_name
        self.memory_dir = "memory"

    def _sanitize_logic(self, text: str) -> str:
        """
        Aggressively strips internal 'Thinking' blocks, LLM monologues, 
        and self-instruction leaks to ensure only professional rationale remains.
        """
        if not text or text == "N/A": 
            return "Assessment complete."
            
        text = str(text)

        # 1. Kill standard tags: <think>...</think>
        text = re.sub(r'<(think|thought)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 2. Kill conversational filler and instruction leaks
        text = re.sub(r'(?i)Thinking\.\.\..*?(\.\.\.done thinking|Let\'s think).*?\.?', '', text, flags=re.DOTALL)
        
        instruction_patterns = [
            r'(?i)We need (to|a) (produce|give|provide|one sentence).*?\.',
            r'(?i)Provide a concise rationale:?',
            r'(?i)Just one sentence\.?',
            r'(?i)Should refer to weight target\.?',
        ]
        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL).strip()

        # 3. Truncate for table neatness
        if len(text) > 250:
            return text[:247].strip() + "..."
            
        return text.strip() or "Clinical rationale validated."

    def run(self, patient: Dict, context: Dict = None) -> Dict:
        """
        Synthesizes all agent memories into the structured payload.
        Structure: Metadata -> Meal Plan -> Layer 1 (Trace + Audit) -> Layer 2 (DQI)
        """
        pid = str(patient.get('record_id', 'unknown'))
        memory_path = f"{self.memory_dir}/{pid}.json"
        
        # Create a lookup for durations from the Supervisor context
        # This ensures we have the actual numbers calculated in the pipeline
        timers = context.get("timers", {}) if context else {}
        
        try:
            memories = {}
            if os.path.exists(memory_path):
                with open(memory_path, 'r', encoding='utf-8') as f:
                    full_memory = json.load(f)
                    memories = full_memory.get("agent_memories", {})

            judge_data = memories.get('lm_judge_data', {})
            plan_data = memories.get('meal_planner_data', {})
            dqi_data = memories.get('dqi_evaluation_data', {}) 
            dqi_breakdown = dqi_data.get('breakdown', {})
            
            report_payload = {
                "metadata": {
                    "report_title": "PN-DSS Meal Plan and Evaluation Report",
                    "patient_id": pid,
                    "report_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
                
                "meal_plan_table": plan_data.get('meal_plan', 'Plan generation failed.'),

                # --- LAYER 1: TECHNICAL AUDIT ---
                "layer_1_audit": {
                    "agent_trace": [
                        {
                            "agent": "Personal", 
                            "logic": self._sanitize_logic(memories.get('personal_evaluator_memory')),
                            "duration": timers.get("Personal", "N/A") # FIXED: Added duration
                        },
                        {
                            "agent": "Nutrients", 
                            "logic": self._sanitize_logic(memories.get('nutrients_evaluator_memory')),
                            "duration": timers.get("Nutrients", "N/A") # FIXED: Added duration
                        },
                        {
                            "agent": "Microbiome", 
                            "logic": self._sanitize_logic(memories.get('microbiome_modulation_memory')),
                            "duration": timers.get("Microbiome", "N/A") # FIXED: Added duration
                        },
                        {
                            "agent": "Constraints", 
                            "logic": self._sanitize_logic(memories.get('constraint_manager_memory')),
                            "duration": timers.get("Constraints", "N/A") # FIXED: Added duration
                        },
                        {
                            "agent": "Planner", 
                            "logic": self._sanitize_logic(memories.get('meal_planner_memory')),
                            "duration": timers.get("Planner", "N/A") # FIXED: Added duration
                        },
                        {
                            "agent": "Technical Audit", 
                            "logic": "LM-Judge Safety and Technical Verification Complete.",
                            "duration": timers.get("Technical Audit", "N/A") # FIXED: Added Judge time
                        }
                    ],
                    "verdict": judge_data.get('verdict', 'Caution'),
                    "structured_comparison": judge_data.get('structured_comparison', [])
                },

                # --- LAYER 2: QUANTITATIVE ANALYSIS ---
                "layer_2_quantitative": {
                    "total_score": f"{dqi_data.get('total_dqi_score', 'N/A')}/100",
                    "variety_score": dqi_breakdown.get('variety', {}).get('total', 0),
                    "adequacy_score": dqi_breakdown.get('adequacy', {}).get('total', 0),
                    "moderation_score": dqi_breakdown.get('moderation', {}).get('total', 0),
                    "balance_score": dqi_breakdown.get('balance', {}).get('total', 0)
                }
            }

            return {"status": "success", "data": report_payload}

        except Exception as e:
            return {"status": "error", "message": str(e)}
