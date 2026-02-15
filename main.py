# main.py

import asyncio
import os
import sys
import time  # Added for UX Performance Tracking
import argparse
from dotenv import load_dotenv

# --- Agent Imports ---
from src.nutrition_supervisor_agent import NutritionSupervisorAgent
from src.agents.personal_evaluator_agent import PersonalEvaluatorAgent
from src.agents.nutrients_evaluator_agent import NutrientsEvaluatorAgent
from src.agents.microbiome_modulation_agent import MicrobiomeModulationAgent
from src.agents.constraint_manager_agent import ConstraintManagerAgent
from src.agents.meal_planner_agent import MealPlannerAgent
from src.agents.report_agent import ReportAgent
from src.agents.lm_judge_agent import LMJudgeAgent

load_dotenv()

async def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="PN-DSS Clinical Pipeline")
    parser.add_argument("--id", type=str, default=None, help="Patient ID") 
    parser.add_argument("--query", type=str, default="Generate a 7-day plan optimized for fiber intake.", help="Clinical Query")
    args = parser.parse_args()

    # 2. Initialize the System (Supervisor is the entry point)
    try:
        pndss_system = NutritionSupervisorAgent(
            personal_evaluator_agent=PersonalEvaluatorAgent(),
            nutrients_evaluator_agent=NutrientsEvaluatorAgent(),
            microbiome_modulation_agent=MicrobiomeModulationAgent(),
            constraint_manager_agent=ConstraintManagerAgent(),
            meal_planner_agent=MealPlannerAgent(),
            report_agent=ReportAgent(),
            lm_judge_agent=LMJudgeAgent()
        )
    except Exception as e:
        print(f"Critical Error: Failed to initialize agents: {e}")
        return
    
    # 3. Stateful ID Check 
    target_id = args.id
    if not target_id:
        if hasattr(pndss_system, 'current_patient_id') and pndss_system.current_patient_id:
            target_id = pndss_system.current_patient_id
            print(f"[SYSTEM] No ID provided. Defaulting to active session: {target_id}")
        else:
            target_id = input("[SYSTEM] No Patient ID found. Please enter Patient ID: ").strip()
            if not target_id:
                print("Error: Patient ID is required.")
                return

    print("\n" + "="*50)
    print("PN-DSS CLINICAL PIPELINE START")
    print(f"Patient ID: {target_id}")
    print(f"Query: {args.query}")
    print("="*50 + "\n")

    # 4. Execution & UX Performance Tracking
    start_time = time.time() 
    try:
        # Pass both ID and Terminal Query to the Supervisor
        result = await pndss_system.supervise(patient_id=target_id, query=args.query)
    except Exception as e:
        print(f"Pipeline Failure: {e}")
        return
    finally:
        total_runtime = round(time.time() - start_time, 2)

    # 5. Handle Results
    # Normalize status check to handle various agent return cases
    status = str(result.get('status', 'ERROR')).upper()
    
    print("\n" + "="*50)
    print(f"FINAL PIPELINE OUTPUT (Total Time: {total_runtime}s)")
    print("="*50)

    if status in ["SUCCESS", "COMPLETED", "APPROVED", "WARNING"]:
        # A. Judge Feedback (Stage 5 - Stateless Assessment)
        judge_data = result.get('judge_result', {})
        verdict = judge_data.get('verdict', 'N/A')
        critique = judge_data.get('critique', 'No critique available.')

        print(f"\n>>> PIPELINE STATUS: {status} (Verdict: {verdict})")
        # Displaying first 300 chars of critique for readability
        print(f"    Judge Critique: {critique[:300]}...")

        # B. PDF Path Check (Supervisor handles file persistence)
        report_path = result.get('report_path') or f"reports/report_{target_id}.pdf"
        if os.path.exists(report_path):
            print(f"\n[SUCCESS] Clinical Report generated: {report_path}")
        else:
            print(f"\n[PARTIAL] Pipeline finished, but PDF file not found at {report_path}")
            
    else:
        print(f"\n[FAILED] System Error: {result.get('message', 'Unknown error')}")

    # 6. Decision Trace (Stage 6 - Response Synthesis)
    print("\n" + "-"*20 + " CLINICAL TRACE " + "-"*20)
    outcomes = result.get('agent_outcomes', [])
    for agent_name, role, rationale in outcomes:
        print(f"-> [{agent_name}] ({role})")
        # Clean rationale for console display
        clean_rationale = str(rationale).replace('\n', ' ')[:150]
        print(f"   Rationale: {clean_rationale}...\n")
    print("-" * 56)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
        sys.exit(0)