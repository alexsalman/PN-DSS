# nutrition agent that creates nutrition plans for patients
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess
import token_1

pat = 20189
r= token_1.get_patient_data(pat)
# print('HTTP Status: ' + str(r.status_code))
age = r[0]['edad_v22']
sex = r[0]['sexo']

dict_keys(['id_personal', 'id_segovia', 'sexo', 'centro', 'fec_entr', 'edad_v22', 'ano_revision_v22', 'hora_inicio_cuestion_v22', 'est_civi_v22', 'poblacion_v22', 'vivi_5an_v22', 'tiem_loc_v22', 'ocup_act_v22', 'trab_act_v22', 'mot_not_v22', 'ocu_act_v22', 'cla_soci_v22', 'ejer_cat_v22', 'cual_die_v22', 'cual_esp_v22', 'demographic_data_complete', 'met_diab_v22', 'diab_pas_v22', 'insulina_v22', 'hat_med_v22', 'med_tens_v22', 'hta_pas_v22', 'col_med_v22', 'cora_med_v22', 'diver_colon_v22', 'col_inter_enf_inflam_v22', 'cron_med_v22', 'cron_cua_v22', 'partos_v22', 'hora_inicio_exp_fisica', 'hernia', 'hernia_tipo', 'peso_v22', 'talla_v22', 'cin_min1_v22', 'cadera1_v22', 'cin_min2_v22', 'cadera2_v22', 'muslo_v22', 'circ_bra_cm_v22', 'manguito_v22', 'pres_obl_v22', 'pres_inf_v22', 'fc1_v22', 'pulso1_v22', 'pas1_v22', 'pad1_v22', 'fc2_v22', 'pulso2_v22', 'pas2_v22', 'pad2_v22', 'fc3_v22', 'pulso3_v22', 'pas3_v22', 'pad3_v22', 'fec_ecg_v22', 'ecg_v22', 'ecg_alte_v22', 'medicam_1_v22', 'duracion_1_v22', 'medicam_2_v22', 'duracion_2_v22', 'medicam_3_v22', 'duracion_3_v22', 'medicam_4_v22', 'duracion_4_v22', 'clinical_data_complete', 'adher_1', 'adher_2', 'adher_3', 'adher_4', 'adher_5', 'adher_6', 'adher_7', 'adher_8', 'adher_9', 'adher_10', 'adher_11', 'adher_12', 'adher_13', 'adher_14', 'puntuacion_14_pts', 'id_ffq', 'hc_g', 'proteins_g', 'fat_g', 'alcohol_g', 'total_energy_kcal', 'hc_porciento', 'proteins_porciento', 'fat_porciento', 'mo_g', 'po_g', 'sa_g', 'mo_porciento', 'po_porciento', 'sa_porciento', 'dietary_fibre_g', 'vit_a_microg', 'vit_d_microg', 'vit_e_mg', 'vit_c_mg', 'vit_b1_mg', 'vit_b2_mg', 'vit_b3_mg', 'vit_b6_mg', 'folic_acid_microg', 'vit_b12_microg', 'ca_mg', 'fe_mg', 'na_mg', 'mg_mg', 'k_mg', 'i_microg', 'se_microg', 'zn_mg', 'p_mg', 'vegetables_g', 'fruits_g', 'legumes_g', 'cereals_g', 'cereals_wholegrain_g', 'cereals_without_potatoes_g', 'dairy_g', 'meats_g', 'olive_oil_g', 'fish_g', 'colest_mg', 'trans_g', 'linoleico_g', 'omega3_g', 'cargluce_g', 'lacteo_enteros_g', 'nutritional_data_complete', 'glb_cap1_v22', 'glb_cap2_v22', 'sog_hora_v22', 'glucosa_2h_v22', 'diabetico_v22', 'analytical_parameters_complete', 'p_euryarchaeota', 'p_thermoplasmatota', 'p_acidobacteriota', 'p_actinobacteriota', 'p_bacteroidota', 'p_bdellovibrionota', 'p_campilobacterota', 'p_chloroflexi', 'p_coprothermobacterota', 'p_cyanobacteria', 'p_dtb120', 'p_desulfobacterota', 'p_elusimicrobiota', 'p_fibrobacterota', 'p_firmicutes', 'p_fusobacteriota', 'p_gemmatimonadota', 'p_halanaerobiaeota', 'p_myxococcota', 'p_patescibacteria', 'p_planctomycetota', 'p_protebacteria', 'p_spirochaetota', 'p_synergistota', 'p_thermotogota', 'p_verrucomicrobiota', 'f_b_ratio', 'gut_microbiota_data_complete'])



# # Load CSV
# df = pd.read_csv("test.csv")

# # Load embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Create embeddings
# texts = df.apply(lambda row: f"{row.name}, {row.age}yo, {row.gender}, {row.weight}kg, {row.height}cm, Goal: {row.goal}, Conditions: {row.medical_conditions}, Prefs: {row.preferences}", axis=1)
# embeddings = model.encode(texts.tolist(), convert_to_numpy=True)

# # Setup FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)

# # Prompt user
# query = input("Enter patient name to generate a nutrition plan: ").strip()
# patient_idx = df.index[df['name'].str.lower() == query.lower()].tolist()

# if not patient_idx:
#     print("Patient not found.")
#     exit()

# # Retrieve patient info
# i = patient_idx[0]
# context = texts[i]

# # Build LLM prompt
# prompt = f"""
# You are a nutritionist. Create a personalized weekly nutrition plan for the following patient:

# {context}

# Make it specific, medically appropriate, and consider preferences.
# """

# # Call local LLM (Ollama)
# print("\nGenerating plan with Mistral...\n")
# result = subprocess.run(
#     ["ollama", "run", "mistral"],
#     input=prompt.encode(),
#     stdout=subprocess.PIPE
# )

# print(result.stdout.decode())