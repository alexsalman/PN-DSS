# src/nutrition_supervisor_agent.py

import json
import os
import datetime
import asyncio
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
    
    memory_path: str = "data/patient_memory.json"
    memory_registry: Dict[str, Any] = field(default_factory=dict)
    current_patient_id: Optional[str] = None 

    def __post_init__(self):
        self._load_memory_from_disk()

    async def supervise(self, patient_id: Any, query: str = None) -> Dict[str, Any]:
        patient_id_str = str(patient_id)
        self.current_patient_id = patient_id_str 
        
        # 1. State Reconciliation
        existing_state = self.memory_registry.get(patient_id_str, {
            "metadata": {"id": patient_id_str, "first_visit": datetime.datetime.now().strftime("%Y-%m-%d")},
            "visit_history": []
        })

        # 2. Data Acquisition
        try:
            patient = self._fetch_patient_data(patient_id_str)
        except Exception as e:
            return {"status": "ERROR", "message": f"Data Acquisition Failed: {e}"}

        categories = self._categorize(patient)
        missing_checklist = {k: (v is None or v == "") for k, v in patient.items()}
        
        # 3. Context Construction
        context = {
            "metadata": {
                "patient_id": patient_id_str, 
                "timestamp": datetime.datetime.now().isoformat(),
                "visit_count": len(existing_state.get("visit_history", [])) + 1
            },
            "history": existing_state, 
            "user_query": query,
            "agent_trace": [], 
            "agent_outcomes": [], 
            "status": "PROCESSING",
            "timers": {} 
        }

        # --- STAGE 3: Reasoning (Stateful) ---
        evaluator_steps = [
            ("Personal Evaluator", self.personal_evaluator_agent),
            ("Nutrients Evaluator", self.nutrients_evaluator_agent),
            ("Microbiome Modulation", self.microbiome_modulation_agent),
            ("Constraint Manager", self.constraint_manager_agent)
        ]
        
        print(f"\n[REASONING PHASE] Executing clinical evaluations...")
        for name, agent in tqdm(evaluator_steps, desc="Clinical Reasoning", unit="agent"):
            if not agent: continue
            
            start_agent = time.time()
            res = agent.run(patient, categories, missing_checklist, context)
            context["timers"][name] = round(time.time() - start_agent, 2)
            
            key = name.lower().replace(" ", "_")
            context[key] = res
            
            clean_summary = self._clean_summary(str(res.get('summary', 'Analysis complete.')))
            context["agent_outcomes"].append([name, "Reasoning", clean_summary])
            # Keep trace for PDF
            context["agent_trace"].append([name, "Reasoning", res.get('summary')])

        # --- STAGE 4: Planning (Stateful) ---
        print(f"\n[PLANNING PHASE] Generating Meal Plan...")
        start_plan = time.time()
        planner_res = self.meal_planner_agent.run(patient, categories, missing_checklist, context)
        context["timers"]["Meal Planner"] = round(time.time() - start_plan, 2)
        context['meal_planner'] = planner_res
        
        context["agent_outcomes"].append([
            "Meal Planner", 
            "Planning", 
            self._clean_summary(planner_res.get('summary', '7-day plan generated.'))
        ])
        context["agent_trace"].append(["Meal Planner", "Planning", planner_res.get('summary')])

        # --- STAGE 5: Evaluation (Stateless LM Judge) ---
        print(f"\n[EVALUATION PHASE] Auditing Plan for Completeness...")
        start_judge = time.time()
        # FIXED: Only passing context (stateless)
        judge_res = self.lm_judge_agent.run(context)
        context["timers"]["LM Judge"] = round(time.time() - start_judge, 2)
        context['lm_judge'] = judge_res
        
        judge_data = judge_res.get('data', {})
        verdict = judge_data.get('verdict', 'Review')
        context["agent_outcomes"].append(["LM Judge", "Evaluation", f"Verdict: {verdict}"])
        context["agent_trace"].append(["LM Judge", "Evaluation", judge_data.get('critique')])

        # --- STAGE 6: Response (Stateless Reporting) ---
        print("\n[RESPONSE PHASE] Finalizing Clinical Report...")
        if self.report_agent:
            context["agent_outcomes"].append(["Report Agent", "Reporting", "Compiling clinical trace into PDF."])
            
            start_report = time.time()
            # FIXED: Only passing patient and context (stateless)
            report_out = self.report_agent.run(patient, context)
            context["timers"]["Report Agent"] = round(time.time() - start_report, 2)
            
            context['report_agent'] = report_out
            context["status"] = "COMPLETED"
            context["judge_result"] = judge_data 
            
            # Update trace with final logic
            context["agent_trace"].append(["Report Agent", "Response", report_out.get('summary')])
            
            # Generate the file using your ReportAgent payload
            filename = self._generate_final_pdf(patient_id_str, context)
            context["report_path"] = filename
            
            self._update_patient_state(patient_id_str, context)

        return context

    def _clean_summary(self, text: str) -> str:
        if not text: return "Analysis complete."
        sentence = text.split('.')[0] + "."
        return (sentence[:97] + "...") if len(sentence) > 100 else sentence

    def _fetch_patient_data(self, pid: str) -> Dict:
        if redcap_connector:
            try:
                raw = redcap_connector.get_patient_data(pid)
                if raw and isinstance(raw, list) and len(raw) > 0:
                    data = raw[0]
                    data['record_id'] = pid
                    return data
            except: pass

        local_path = f"data/{pid}.json"
        if os.path.exists(local_path):
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        raise Exception(f"Patient {pid} not found. Ensure data/{pid}.json exists.")

    def _update_patient_state(self, pid: str, ctx: Dict):
        current_visit = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": ctx.get('user_query'),
            "verdict": ctx.get('judge_result', {}).get('verdict'),
            "execution_times": ctx.get('timers'),
            "report_file": ctx.get('report_path')
        }

        if pid not in self.memory_registry:
            self.memory_registry[pid] = {
                "metadata": {"id": pid, "first_visit": datetime.datetime.now().strftime("%Y-%m-%d")},
                "visit_history": []
            }
        
        self.memory_registry[pid]["visit_history"].append(current_visit)

        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory_registry, f, indent=4)

    def _categorize(self, patient: Dict) -> Dict:
        try:
            from src.utils.field_mapping import PATIENT_FIELD_CATEGORIES
            cat = {c: {} for c in PATIENT_FIELD_CATEGORIES}
            cat["uncategorized"] = {}
            mapping = {f: c for c, fields in PATIENT_FIELD_CATEGORIES.items() for f in fields}
            for k, v in patient.items():
                cat[mapping.get(k, "uncategorized")][k] = v
            return cat
        except:
            return {"personal": patient, "uncategorized": patient}

    def _generate_final_pdf(self, pid: str, ctx: Dict) -> str:
        # Use the payload constructed by the stateless ReportAgent
        report_payload = ctx.get('report_agent', {}).get('data', {})
        
        os.makedirs("reports", exist_ok=True)
        filename = f"reports/report_{pid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        # This now matches your save_to_pdf(filename, report_payload) version
        save_to_pdf(filename, report_payload)
        return filename

    def _load_memory_from_disk(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    self.memory_registry = json.load(f)
            except: 
                self.memory_registry = {}