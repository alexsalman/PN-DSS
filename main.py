# main.py

import asyncio
import os
import sys
import time
import argparse
from pathlib import Path
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
    # 1. Blackboard Initialisation
    memory_dir = Path("memory")
    if not memory_dir.exists():
        memory_dir.mkdir(parents=True)
        print("[SYSTEM] Created 'memory' directory for patient state files.")

    # 2. Parse Arguments
    parser = argparse.ArgumentParser(description="PN-DSS Clinical Pipeline")
    parser.add_argument("--query", type=str, help="Full command (e.g., 'ID 20189 7-day high fiber plan')")
    args = parser.parse_args()

    # 3. Agent Initialization
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
    
    # 4. Input Parsing & Dynamic Input Handling
    raw_command = args.query or input("[SYSTEM] Enter Full Clinical Query: ").strip()
    if not raw_command:
        print("Error: A query is required to proceed.")
        return

    print("\n" + "="*50)
    print("PN-DSS CLINICAL PIPELINE START")
    print(f"Input Command: {raw_command}")
    print("="*50 + "\n")

    # 5. Pipeline Launch & Execution with High-Level Timer
    start_time = time.time() 
    try:
        # The result payload should now include 'total_runtime' for the PDF
        result = await pndss_system.supervise(query=raw_command)
    except Exception as e:
        print(f"Pipeline Failure: {e}")
        return
    finally:
        # Calculate wall-clock time for the entire async operation
        total_runtime = round(time.time() - start_time, 2)

    # 6. Multi-Layer Outcome Verification
    status = str(result.get('status', 'ERROR')).upper()
    
    print("\n" + "="*50)
    print(f"FINAL PIPELINE OUTPUT")
    print(f"Total Execution Time: {total_runtime}s")
    print("="*50)

    if status == "COMPLETED":
        # Layer 1 Audit: Verdict
        judge_data = result.get('judge_result', {})
        print(f"\n>>> PIPELINE STATUS: {status} (Verdict: {judge_data.get('verdict', 'N/A')})")
        
        # File Persistence Check
        report_path = result.get('report_path')
        if report_path and os.path.exists(report_path):
            print(f"[SUCCESS] Clinical Report generated: {report_path}")
        else:
            print(f"\n[PARTIAL] Pipeline finished, but PDF file not found at {report_path}")
            
    else:
        print(f"\n[FAILED] System Error: {result.get('message', 'Unknown error')}")

    # 7. Layer 3 Trace (Clinical Decision Log + Latency Audit)
    print("\n" + "-"*20 + " CLINICAL TRACE " + "-"*20)
    outcomes = result.get('agent_outcomes', [])
    
    # Unpack agent results, checking for the fourth timing element
    for agent_data in outcomes:
        # Handle varying tuple lengths gracefully
        agent_name = agent_data[0]
        role = agent_data[1]
        rationale = agent_data[2]
        duration = agent_data[3] if len(agent_data) > 3 else "N/A"
        
        # Check for thinking leaks (Coherence Audit)
        leak_flag = "[!] LEAK" if "Thinking..." in str(rationale) else "[OK]"
        
        # Display Agent Name, Status, and individual Latency
        print(f"-> [{agent_name}] ({role}) {leak_flag} | Latency: {duration}s")
        # Truncate rationale for cleaner terminal view
        print(f"   Rationale: {str(rationale)[:150]}...\n")
        
    print("-" * 25 + f" End of Trace ({total_runtime}s) " + "-" * 25)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
        sys.exit(0)
