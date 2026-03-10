# src/nutrition_supervisor_agent.py

import json
import os
import datetime
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from tqdm import tqdm

# --- Agent Imports ---
from src.agents.personal_evaluator_agent import PersonalEvaluatorAgent
from src.agents.nutrients_evaluator_agent import NutrientsEvaluatorAgent
from src.agents.microbiome_modulation_agent import MicrobiomeModulationAgent
from src.agents.constraint_manager_agent import ConstraintManagerAgent
from src.agents.meal_planner_agent import MealPlannerAgent
from src.agents.report_agent import ReportAgent
from src.agents.lm_judge_agent import LMJudgeAgent

# --- Utility Imports ---
from src.utils.dqi_calculator import DQICalculator
try:
    import src.utils.redcap_connector as redcap_connector
except ImportError:
    redcap_connector = None

from src.utils.pdf_generator import save_to_pdf

@dataclass
class NutritionSupervisorAgent:
    personal_evaluator_agent: Optional[PersonalEvaluatorAgent] = None
    nutrients_evaluator_agent: Optional[NutrientsEvaluatorAgent] = None
    microbiome_modulation_agent: Optional[MicrobiomeModulationAgent] = None
    constraint_manager_agent: Optional[ConstraintManagerAgent] = None
    meal_planner_agent: Optional[MealPlannerAgent] = None
    report_agent: Optional[ReportAgent] = None
    lm_judge_agent: Optional[LMJudgeAgent] = None
    
    memory_dir: str = "memory"
    dqi_calculator: DQICalculator = field(default_factory=DQICalculator)

    def __post_init__(self):
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

    async def supervise(self, query: str = "") -> Dict[str, Any]:
        pipeline_start = time.time()
        
        # 1. PARSE QUERY
        id_match = re.search(r'(?:ID\s*)?(\d{4,})', query, re.IGNORECASE)
        days_match = re.search(r'(\d+)\s*day', query.lower())
        pid = id_match.group(1) if id_match else None
        plan_duration = days_match.group(1) if days_match else "7"

        if not pid:
            return {"status": "ERROR", "message": "No valid Patient ID found."}

        # 2. BLACKBOARD RETRIEVAL
        patient = self._get_patient_state(pid)

        # 3. CONTEXT INITIALIZATION
        categories = self._categorize(patient)
        missing_checklist = {k: (v is None or v == "") for k, v in patient.items()}
        
        context = {
            "metadata": {
                "patient_id": pid, 
                "plan_duration": plan_duration,
                "report_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "visit_count": patient.get("metadata", {}).get("visit_count", 0) + 1
            },
            "user_query": query,
            "agent_outcomes": [], 
            "timers": {}
        }

        # 4. REASONING PHASE
        evaluator_steps = [
            ("Personal", self.personal_evaluator_agent, "personal_evaluator_memory"),
            ("Nutrients", self.nutrients_evaluator_agent, "nutrients_evaluator_memory"),
            ("Microbiome", self.microbiome_modulation_agent, "microbiome_modulation_memory"),
            ("Constraints", self.constraint_manager_agent, "constraint_manager_memory")
        ]

        print(f"\n[REASONING PHASE] Clinical Evaluation for Patient {pid}")
        for name, agent, memory_key in tqdm(evaluator_steps, desc="Clinical Reasoning"):
            if not agent: continue
            step_start = time.time()
            agent.run(patient, categories, missing_checklist, context)
            dur = round(time.time() - step_start, 2)
            
            context["timers"][name] = dur
            final_memory = self._read_latest_memory(pid)
            rationale = final_memory.get("agent_memories", {}).get(memory_key, "Complete.")
            context["agent_outcomes"].append([name, "Reasoning", rationale, dur])

        # 5. PLANNING PHASE
        start_plan = time.time()
        self.meal_planner_agent.run(patient, categories, missing_checklist, context)
        plan_dur = round(time.time() - start_plan, 2)
        context["timers"]["Planner"] = plan_dur
        
        planner_mem = self._read_latest_memory(pid).get("agent_memories", {})
        planner_data = planner_mem.get("meal_planner_data", {})
        context["agent_outcomes"].append(["Planner", "Planning", "Plan generated.", plan_dur])

        # 6. EVALUATION PHASE
        # --- DQI Calculator ---
        start_dqi = time.time()
        dqi_results = self.dqi_calculator.get_dqi_report(planner_data.get('meal_plan', ''), planner_data.get('clinical_targets', {}))
        self._save_to_blackboard(pid, "dqi_evaluation_data", dqi_results)
        context["timers"]["DQI Calculator"] = round(time.time() - start_dqi, 2)
        
        # --- LAYER 1: JUDGE (The Technical Audit) ---
        start_judge = time.time()
        judge_res = self.lm_judge_agent.run(context)
        judge_dur = round(time.time() - start_judge, 2)
        
        context["judge_result"] = judge_res.get("data", {})
        context["timers"]["Technical Audit"] = judge_dur
        # Inject the Judge as a visible outcome for the Audit Table
        context["agent_outcomes"].append(["Technical Audit", "Safety", "Validation Complete.", judge_dur])

        # 7. RESPONSE PHASE & PDF GENERATION
        self._update_metadata(pid, patient, context)
        report_out = self.report_agent.run(patient, context)
        report_payload = report_out.get('data', {})

        # --- BRIDGE LOGIC: FIXING "N/A" AND MISSING JUDGE TIME ---
        if 'layer_1_audit' in report_payload:
            # Create map from our context timers
            duration_map = {outcome[0]: outcome[3] for outcome in context["agent_outcomes"] if len(outcome) > 3}
            trace = report_payload['layer_1_audit'].get('agent_trace', [])
            updated_trace = []
            seen_agents = set()

            for entry in trace:
                name = entry.get('agent')
                if name in duration_map:
                    entry['duration'] = duration_map[name] # Replace N/A
                    seen_agents.add(name)
                updated_trace.append(entry)
            
            # Ensure Technical Audit is in the list
            if "Technical Audit" not in seen_agents:
                updated_trace.append({
                    "agent": "Technical Audit",
                    "logic": "LM-Judge Safety and Technical Verification Complete.",
                    "duration": judge_dur
                })
            report_payload['layer_1_audit']['agent_trace'] = updated_trace

        # Update global latency in header
        total_time = round(time.time() - pipeline_start, 2)
        report_payload.setdefault('metadata', {})['total_generation_time'] = total_time

        os.makedirs("reports", exist_ok=True)
        filename = f"reports/report_{pid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        save_to_pdf(filename, report_payload)
        
        return {"status": "COMPLETED", "report_path": filename, "total_runtime": total_time}

    def _save_to_blackboard(self, pid: str, key: str, data: Dict):
        path = f"{self.memory_dir}/{pid}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                mem = json.load(f)
            mem.setdefault("agent_memories", {})[key] = data
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(mem, f, indent=4)

    def _read_latest_memory(self, pid: str) -> Dict:
        path = f"{self.memory_dir}/{pid}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _get_patient_state(self, pid: str) -> Dict:
        path = f"{self.memory_dir}/{pid}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        if redcap_connector:
            raw = redcap_connector.get_patient_data(pid)
            if raw and len(raw) > 0:
                data = raw[0]
                data['record_id'] = pid
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                return data
        raise Exception(f"Patient {pid} not found in memory.")

    def _update_metadata(self, pid: str, patient_data: Dict, ctx: Dict):
        path = f"{self.memory_dir}/{pid}.json"
        current_state = self._read_latest_memory(pid)
        current_state.setdefault("metadata", {})["last_visit"] = datetime.datetime.now().isoformat()
        current_state["metadata"]["visit_count"] = ctx["metadata"]["visit_count"]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(current_state, f, indent=4)

    def _categorize(self, patient: Dict) -> Dict:
        return {"personal": patient, "clinical": patient}
