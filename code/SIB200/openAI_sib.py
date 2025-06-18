import os
import openai
import json
import pandas as pd
from datasets import load_dataset

from openai import OpenAI
client = OpenAI()

import argparse

# results_folder = '../results/SIB200/sib_results_gpt35'
# results_reply_folder = '../results/SIB200/sib_replies_gpt35'
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

prompt_sib="You are an assistant able to classify topics in texts. \n\nGiven the categories science/technology, travel, politics, sports, health, entertainment, or geography; what is the topic of the {} statement below? Return only the category. \n\ntext: {} \ncategory:\n\n"

import re
categories = ['science/technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography']
def get_category(prompt):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ])
    reply = completion.choices[0].message.content
    reply = reply.lower()
    # print(reply)
    category = ''
    if 'science' in reply or 'technology' in reply:
        category = categories[0]
    elif 'travel' in reply:
        category = categories[1]
    elif 'politics' in reply:
        category = categories[2]
    elif 'sports' in reply:
        category = categories[3]
    elif 'health' in reply:
        category = categories[4]
    elif 'entertainment' in reply:
        category = categories[5]
    elif 'geography' in reply:
        category = categories[6]
    else:
        category = 'invalid'
    return reply, category

df = pd.read_csv('csvs/SIB200/sib_gpt4o_results_april2025.csv')
code_to_name_map = df.set_index('language code')['Language name'].to_dict()

from tqdm import tqdm
result_categories = {}
result_accuracies = {}
gpt_replies = {}

for language_code in languages_to_run:
    completed = ['']
    if language_code in completed:
        continue
    print(language_code)
    accurate = 0

    language_name = code_to_name_map[language_code]
    result_categories[language_code]=[]
    gpt_replies[language_code] = []
    dataset = load_dataset("Davlan/sib200", language_code) 

    size = len(dataset['test'])
    for i in tqdm(range(size)):
        text = dataset['test'][i]['text']
        prompt = prompt_sib.format(language_name, text)
        
        reply, category = get_category(prompt)
        gpt_replies[language_code].append(reply)

        if category == '':
            result_categories[language_code].append('invalid')
            continue
            
        result_categories[language_code].append(category)
        if category == dataset['test'][i]['category']:
            accurate+=1
    result_accuracies[language_code] = accurate

    df = pd.DataFrame({
    "text": dataset['test']['text'],
    "gpt_label": result_categories[language_code]
    })

    # Save to CSV
    df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)

    df2 = pd.DataFrame({
    "text": dataset['test']['question'],
    "gpt_reply": gpt_replies[language_code]
    })

    # Save to CSV
    df2.to_csv("{}/{}.csv".format(results_reply_folder,language_code), index=False)



import json

# Save to a JSON file
with open("result_accuracies_prompt2_gpt35.json", "w", encoding="utf-8") as f:
    json.dump(result_accuracies, f, ensure_ascii=False, indent=4)

df = pd.DataFrame({
    "Language name": [code_to_name_map[language_code] for language_code in sib_languages],
    "Language code": sib_languages,
    "Accuracy": [round(result_accuracies[language_code]*100/204, 1) for language_code in sib_languages]
})

# Save to CSV
df.to_csv("{}/{}.csv".format(results_csv_folder,'sib_gpt35_results_april2025.csv'), index=False)

#### This below is to make the compiled csvs with all results ### 

# language_mapping = pd.read_csv('csvs/language_mapping.csv')
# language_mapping_index = language_mapping.set_index('FLORES-200 code')['Language'].to_dict()
# flores_languages = list(language_mapping_index.keys())

# # Convert to DataFrame
# all_languages = list(language_mapping_index.keys()) #code

# all_languages2 = list(language_mapping_index.values()) #name

# import bisect
# # New elements to insert
# new_key = "nqo_Nkoo"
# new_value = "N'ko"

# # Find the correct index to insert while maintaining sorted order
# index = bisect.bisect(all_languages, new_key)

# # Insert in both lists at the correct position
# all_languages.insert(index, new_key)
# all_languages2.insert(index, new_value)

# import json
# with open('result_accuracies_prompt2.json', "r") as f:
#     result_accuracies2 = json.load(f)

# import pandas as pd
# flores_df = pd.read_csv('csvs/Flores/flores_gpt4o_results_april_2025.csv')
# old_results_df = pd.read_csv('csvs/old_results.csv')

# gpt_4_scores = old_results_df.set_index('language_code')['gpt-4'].to_dict()
# gpt_4_array = []
# for lang in all_languages:
#     if lang in gpt_4_scores.keys():
#         gpt_4_array.append(gpt_4_scores[lang])
#     else:
#         gpt_4_array.append('')
        
# gpt_3_scores = old_results_df.set_index('language_code')['gpt-3.5'].to_dict()
# gpt_3_array = []
# for lang in all_languages:
#     if lang in gpt_3_scores.keys():
#         gpt_3_array.append(gpt_3_scores[lang])
#     else:
#         gpt_3_array.append('')

# lang_family = old_results_df.set_index('language_code')['language_family'].to_dict()
# lang_family_array = []
# for lang in all_languages:
#     if lang in lang_family.keys():
#         lang_family_array.append(lang_family[lang])
#     else:
#         lang_family_array.append('')
        
# flores_xx_eng_scores = flores_df.set_index('FLORES-200 code')['xx - English'].to_dict()
# flores_xx_eng_array = []
# for lang in all_languages:
#     if lang in flores_xx_eng_scores.keys():
#         flores_xx_eng_array.append(flores_xx_eng_scores[lang])
#     else:
#         flores_xx_eng_array.append('')

# flores_eng_xx_scores = flores_df.set_index('FLORES-200 code')['English - xx'].to_dict()
# flores_eng_xx_array = []
# for lang in all_languages:
#     if lang in flores_eng_xx_scores.keys():
#         flores_eng_xx_array.append(flores_eng_xx_scores[lang])
#     else:
#         flores_eng_xx_array.append('')


# result_df = pd.DataFrame({
#     "language_name": all_languages2,
#     "language_code": all_languages,
#     "language_family": lang_family_array,
#     "SIB gpt-3.5 (old)": gpt_3_array,
#     "SIB gpt-4 (old)": gpt_4_array,
#     "SIB gpt-3.5-turbo-1106 (Apr 2025)": [round(result_accuracies[lang]*100/204, 1) for lang in all_languages],
#     "SIB gpt-4o (Apr 2025)": [round(result_accuracies2[lang]*100/204, 1) for lang in all_languages],
#     "Flores xx-eng gpt-4o (Apr 2025)": flores_xx_eng_array,
#     "Flores eng-xx gpt-4o (Apr 2025)": flores_eng_xx_array

# })
# result_df.to_csv('csvs/compiled/overall_results.csv',index=False)

# df2 = pd.read_csv('overall_results.csv')
# z=df2.groupby('language_family') \
#        .agg({'language_name':'size', 'SIB gpt-3.5 (old)':'mean','SIB gpt-4 (old)':'mean','SIB gpt-3.5-turbo-1106 (Apr 2025)':'mean','SIB gpt-4o (Apr 2025)':'mean','Flores xx-eng gpt-4o (Apr 2025)':'mean','Flores eng-xx gpt-4o (Apr 2025)':'mean'} ) \
#        .rename(columns={'language_name':'count'}) \
#        .sort_values(by=['count'], ascending=False) \
#        .reset_index().round(1)
# z.to_csv('csvs/compiled_by_language_family/overall_by_family.csv',index=False)
