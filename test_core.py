# test_core.py
from agent_core import NutritionAgentCore

agent = NutritionAgentCore()

sample_patient = {
    "record_id": "0001",
    "edad_v22": 59,
    "sexo": "2",
    "peso_v22": 86.1,
    "talla_v22": 165,
    "waist": 100,
    "hip": 105,
    "total_energy_kcal": 1800,
    "proteins_g": 70,
    "fat_g": 65,
    "dietary_fibre_g": 9.6,
    "activity": "moderate",
    "med_diet_score": 6,
    "p_firmicutes": 60,
    "p_bacteroidota": 40,
    "diabetico_v22": "1",
    "hta": "1",
    "cholesterol": "0",
    "cvd": "0",
    "ibd": "0"
}

result = agent.evaluate_patient(sample_patient)

from pprint import pprint
pprint(result)