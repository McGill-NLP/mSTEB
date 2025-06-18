import os
import openai
import json
import pandas as pd
from datasets import load_dataset

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

import argparse

# results_folder = '../results/SIB14/sib14_results_gemini2'
# results_reply_folder = '../results/SIB14/sib14_replies_gemini2'
# results_csv_folder = '../csvs/SIB200'

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True)
parser.add_argument("--results_reply_folder", type=str, required=True)
parser.add_argument("--results_csv_folder", type=str, required=True)

args = parser.parse_args()

results_folder = args.results_folder
results_reply_folder = args.results_reply_folder
results_csv_folder = args.results_csv_folder

sib_languages = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'als_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'arb_Latn', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gaz_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khk_Cyrl', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kmr_Latn', 'knc_Arab', 'knc_Latn', 'kon_Latn', 'kor_Hang', 'lao_Laoo', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'lvs_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Arab', 'min_Latn', 'mkd_Cyrl', 'mlt_Latn', 'mni_Beng', 'mos_Latn', 'mri_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nqo_Nkoo', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pbt_Arab', 'pes_Arab', 'plt_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Olck', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'taq_Latn', 'taq_Tfng', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zsm_Latn', 'zul_Latn']
languages_to_run = sib_languages[0:205]

prompt_sib="You are an assistant able to classify topics in texts. \n\nGiven the categories travel, politics, science, sports, technology, health, nature, entertainment, geography, business, disasters, crime, education, religion; what is the topic of the {} statement below? Return only the category. \n\ntext: {} \ncategory:\n\n"

import re
import time

categories = ["travel", "politics", "science", "sports", "technology", "health", "nature", "entertainment", "geography",
"business", "disasters", "crime", "education", "religion"]

def get_category(prompt, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                    )
            reply = response.text.lower()
            # print(reply)
            category = ''
            if 'travel' in reply:
                category = categories[0]
            elif 'politics' in reply:
                category = categories[1]
            elif 'science' in reply:
                category = categories[2]
            elif 'sports' in reply:
                category = categories[3]
            elif 'technology' in reply:
                category = categories[4]
            elif 'health' in reply:
                category = categories[5]
            elif 'nature' in reply:
                category = categories[6]
            elif 'entertainment' in reply:
                category = categories[7]
            elif 'geography' in reply:
                category = categories[8]
            elif 'business' in reply:
                category = categories[9]
            elif 'disasters' in reply:
                category = categories[10]
            elif 'crime' in reply:
                category = categories[11]
            elif 'education' in reply:
                category = categories[12]
            elif 'religion' in reply:
                category = categories[13]
            else:
                category = 'invalid'
            return reply, category
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Returning failure label.")
                return "", 'invalid'

df = pd.read_csv('csvs/SIB200/sib_gpt4o_results_april2025.csv')
code_to_name_map = df.set_index('language code')['Language name'].to_dict()


from tqdm import tqdm
result_categories = {}
result_accuracies = {}
gpt_replies = {}
result_inaccuracies = {}

for language_code in languages_to_run:
    completed = []
    if language_code in completed:
        continue
    print(language_code)
    accurate = 0
    errors = 0
    language_name = code_to_name_map[language_code]
    
    result_categories[language_code]=[]
    gpt_replies[language_code] = []
    dataset = load_dataset("Davlan/sib200_14classes", language_code) 

    size = len(dataset['test'])
    for i in tqdm(range(size)): #length of devtest
        text = dataset['test'][i]['text']
        prompt = prompt_sib.format(language_name, text)
        reply, category = get_category(prompt)
        gpt_replies[language_code].append(reply)
        result_categories[language_code].append(category)
        if category == dataset['test'][i]['category']:
            accurate+=1
        if category == 'invalid':
            errors+=1
    result_accuracies[language_code] = accurate
    result_inaccuracies[language_code] = errors

    df = pd.DataFrame({
    "text": dataset['test']['text'],
    "gemini_label": result_categories[language_code]
    })

    # Save to CSV
    df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)

    df2 = pd.DataFrame({
    "text": dataset['test']['text'],
    "gemini_reply": gpt_replies[language_code]
    })

    # Save to CSV
    df2.to_csv("{}/{}.csv".format(results_reply_folder,language_code), index=False)

    print(accurate)

print(result_accuracies)


import json

# Save to a JSON file
with open("result_accuracies_sib14_gemini2.json", "w", encoding="utf-8") as f:
    json.dump(result_accuracies, f, ensure_ascii=False, indent=4)

df = pd.DataFrame({
    "Language name": [code_to_name_map[language_code] for language_code in sib_languages],
    "Language code": sib_languages,
    "Accuracy": [round(result_accuracies[language_code]*100/1225, 1) for language_code in sib_languages]
})

# Save to CSV
df.to_csv("{}/{}.csv".format(results_csv_folder,'sib14_gemini2_results.csv'), index=False)
