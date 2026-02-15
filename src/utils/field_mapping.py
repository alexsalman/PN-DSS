# src/utils/field_mapping.py

from typing import Dict, Any, Tuple

PATIENT_FIELD_CATEGORIES = {
    "personal": [
        "id_personal", "id_segovia", "sexo", "centro", "fec_entr", "edad_v22", 
        "ano_revision_v22", "est_civi_v22", "poblacion_v22", "vivi_5an_v22", 
        "tiem_loc_v22", "ocup_act_v22", "trab_act_v22", "cla_soci_v22", 
        "ejer_cat_v22", "demographic_data_complete", "peso_v22", "talla_v22"
    ],
    "nutrients": [
        "hc_g", "proteins_g", "fat_g", "alcohol_g", "total_energy_kcal", 
        "hc_porciento", "proteins_porciento", "fat_porciento", "mo_g", "po_g", 
        "sa_g", "mo_porciento", "po_porciento", "sa_porciento", "dietary_fibre_g", 
        "vit_a_microg", "vit_d_microg", "vit_e_mg", "vit_c_mg", "vit_b1_mg", 
        "vit_b2_mg", "vit_b3_mg", "vit_b6_mg", "folic_acid_microg", "vit_b12_microg", 
        "ca_mg", "fe_mg", "na_mg", "mg_mg", "k_mg", "i_microg", "se_microg", 
        "zn_mg", "p_mg", "vegetables_g", "fruits_g", "legumes_g", "cereals_g", 
        "cereals_wholegrain_g", "cereals_without_potatoes_g", "dairy_g", "meats_g", 
        "olive_oil_g", "fish_g", "colest_mg", "trans_g", "linoleico_g", "omega3_g", 
        "cargluce_g", "lacteo_enteros_g", "nutritional_data_complete", 
        "adher_1", "adher_2", "adher_3", "adher_4", "adher_5", "adher_6", 
        "adher_7", "adher_8", "adher_9", "adher_10", "adher_11", "adher_12", 
        "adher_13", "adher_14", "puntuacion_14_pts", "id_ffq"
    ],
    "microbiome": [
        "p_euryarchaeota", "p_thermoplasmatota", "p_acidobacteriota", 
        "p_actinobacteriota", "p_bacteroidota", "p_bdellovibrionota", 
        "p_campilobacterota", "p_chloroflexi", "p_coprothermobacterota", 
        "p_cyanobacteria", "p_dtb120", "p_desulfobacterota", "p_elusimicrobiota", 
        "p_fibrobacterota", "p_firmicutes", "p_fusobacteriota", "p_gemmatimonadota", 
        "p_halanaerobiaeota", "p_myxococcota", "p_patescibacteria", 
        "p_planctomycetota", "p_protebacteria", "p_spirochaetota", 
        "p_synergistota", "p_thermotogota", "p_verrucomicrobiota", "f_b_ratio", 
        "gut_microbiota_data_complete"
    ],
    "constraints": [
        "met_diab_v22", "diab_pas_v22", "insulina_v22", "hat_med_v22", 
        "med_tens_v22", "hta_pas_v22", "col_med_v22", "cora_med_v22", 
        "diver_colon_v22", "col_inter_enf_inflam_v22", "cron_med_v22", 
        "cron_cua_v22", "hernia", "hernia_tipo", "medicam_1_v22", 
        "medicam_2_v22", "medicam_3_v22", "medicam_4_v22", "clinical_data_complete",
        "pas1_v22", "pad1_v22", "pas2_v22", "pad2_v22", "pas3_v22", "pad3_v22",
        "glb_cap1_v22", "glb_cap2_v22", "glucosa_2h_v22", "diabetico_v22"
    ]
}

def _is_missing(value) -> bool:
    """Helper to check if a value is effectively null."""
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return False
    val_str = str(value).strip().lower()
    return val_str in ["", "nan", "none", "n/a", "[empty]", "null"]

def map_categories(patient_record: Dict[str, Any]) -> Tuple[Dict[str, Dict], Dict[str, list]]:
    """
    Takes a flat patient record dictionary and organizes it into:
    1. categories: { 'nutrients': { 'hc_g': 50, ... }, ... }
    2. missing: { 'nutrients': ['vit_c_mg'], ... }
    """
    categories = {}
    missing = {}

    for category_name, fields in PATIENT_FIELD_CATEGORIES.items():
        categories[category_name] = {}
        missing[category_name] = []
        
        for field in fields:
            # We check if the field exists in the record
            val = patient_record.get(field)
            
            if _is_missing(val):
                missing[category_name].append(field)
            else:
                categories[category_name][field] = val
                
    return categories, missing