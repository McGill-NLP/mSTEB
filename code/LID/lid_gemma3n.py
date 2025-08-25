import os
import openai
import json
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("facebook/flores", "all")  # Loads all languages


import argparse

# results_folder = 'results/LID/lid_predicted_gemma3n'
# results_reply_folder = 'results/LID/lid_replies_gemmma3n'
# python code/LID/lid_gemma3n.py   --results_folder='results/LID/lid_predicted_gemma3n'   --results_reply_folder='results/LID/lid_replies_gemma3n'

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True)
parser.add_argument("--results_reply_folder", type=str, required=True)

args = parser.parse_args()

results_folder = args.results_folder
results_reply_folder = args.results_reply_folder

language_mapping = pd.read_csv('csvs/language_mapping.csv')
language_mapping_index = language_mapping.set_index('FLORES-200 code')['Language'].to_dict()
all_languages = list(language_mapping_index.keys())

flores_l = {'afr_Latn': 'Afrikaans',
 'amh_Ethi': 'Amharic',
 'arb_Arab': 'Arabic',
 'asm_Beng': 'Assamese',
 'ast_Latn': 'Asturian',
 'azj_Latn': 'Azerbaijani',
 'bel_Cyrl': 'Belarusian',
 'ben_Beng': 'Bengali',
 'bos_Latn': 'Bosnian',
 'bul_Cyrl': 'Bulgarian',
 'cat_Latn': 'Catalan',
 'ceb_Latn': 'Cebuano',
 'ces_Latn': 'Czech',
 'ckb_Arab': 'Kurdish',
 'cym_Latn': 'Welsh',
 'dan_Latn': 'Danish',
 'deu_Latn': 'German',
 'ell_Grek': 'Greek',
 'eng_Latn': 'English',
 'est_Latn': 'Estonian',
 'fin_Latn': 'Finnish',
 'fra_Latn': 'French',
 'gaz_Latn': 'Oromo',
 'gle_Latn': 'Irish',
 'glg_Latn': 'Galician',
 'guj_Gujr': 'Gujarati',
 'hau_Latn': 'Hausa',
 'heb_Hebr': 'Hebrew',
 'hin_Deva': 'Hindi',
 'hrv_Latn': 'Croatian',
 'hun_Latn': 'Hungarian',
 'hye_Armn': 'Armenian',
 'ibo_Latn': 'Igbo',
 'ind_Latn': 'Indonesian',
 'isl_Latn': 'Icelandic',
 'ita_Latn': 'Italian',
 'jav_Latn': 'Javanese',
 'jpn_Jpan': 'Japanese',
 'kam_Latn': 'Kamba',
 'kan_Knda': 'Kannada',
 'kat_Geor': 'Georgian',
 'kaz_Cyrl': 'Kazakh',
 'kea_Latn': 'Kabuverdianu',
 'khk_Cyrl': 'Mongolian',
 'khm_Khmr': 'Khmer',
 'kor_Hang': 'Korean',
 'lao_Laoo': 'Lao',
 'lin_Latn': 'Lingala',
 'lit_Latn': 'Lithuanian',
 'ltz_Latn': 'Luxembourgish',
 'lug_Latn': 'Ganda',
 'luo_Latn': 'Luo',
 'lvs_Latn': 'Latvian',
 'mal_Mlym': 'Malayalam',
 'mar_Deva': 'Marathi',
 'mkd_Cyrl': 'Macedonian',
 'mlt_Latn': 'Maltese',
 'mri_Latn': 'Maori',
 'mya_Mymr': 'Burmese',
 'nld_Latn': 'Dutch',
 'nob_Latn': 'Norwegian',
 'npi_Deva': 'Nepali',
 'nso_Latn': 'Sotho',
 'nya_Latn': 'Nyanja',
 'oci_Latn': 'Occitan',
 'ory_Orya': 'Odia',
 'pan_Guru': 'Punjabi',
 'pbt_Arab': 'Pashto',
 'pes_Arab': 'Persian',
 'pol_Latn': 'Polish',
 'por_Latn': 'Portuguese',
 'ron_Latn': 'Romanian',
 'rus_Cyrl': 'Russian',
 'slk_Latn': 'Slovak',
 'slv_Latn': 'Slovenian',
 'smo_Latn': 'Samoan',
 'sna_Latn': 'Shona',
 'snd_Arab': 'Sindhi',
 'som_Latn': 'Somali',
 'spa_Latn': 'Spanish',
 'srp_Cyrl': 'Serbian',
 'swe_Latn': 'Swedish',
 'swh_Latn': 'Swahili',
 'tam_Taml': 'Tamil',
 'tel_Telu': 'Telugu',
 'tgk_Cyrl': 'Tajik',
 'tgl_Latn': 'Tagalog',
 'tha_Thai': 'Thai',
 'tur_Latn': 'Turkish',
 'ukr_Cyrl': 'Ukrainian',
 'umb_Latn': 'Umbundu',
 'urd_Arab': 'Urdu',
 'uzn_Latn': 'Uzbek',
 'vie_Latn': 'Vietnamese',
 'wol_Latn': 'Wolof',
 'xho_Latn': 'Xhosa',
 'yor_Latn': 'Yoruba',
 'zho_Hans': 'Chinese',
 'zho_Hant': 'Chinese',
 'zsm_Latn': 'Malay',
 'zul_Latn': 'Zulu'}


remaining_l = {'ace_Arab': 'Acehnese',
 'ace_Latn': 'Acehnese',
 'acm_Arab': 'Arabic',
 'acq_Arab': 'Arabic',
 'aeb_Arab': 'Arabic',
 'ajp_Arab': 'Arabic',
 'aka_Latn': 'Akan',
 'als_Latn': 'Albanian',
 'apc_Arab': 'Arabic',
 'arb_Latn': 'Arabic',
 'ars_Arab': 'Arabic',
 'ary_Arab': 'Arabic',
 'arz_Arab': 'Arabic',
 'awa_Deva': 'Awadhi',
 'ayr_Latn': 'Aymara',
 'azb_Arab': 'Azerbaijani',
 'bak_Cyrl': 'Bashkir',
 'bam_Latn': 'Bambara',
 'ban_Latn': 'Balinese',
 'bem_Latn': 'Bemba',
 'bho_Deva': 'Bhojpuri',
 'bjn_Arab': 'Banjar',
 'bjn_Latn': 'Banjar',
 'bod_Tibt': 'Tibetan',
 'bug_Latn': 'Buginese',
 'cjk_Latn': 'Chokwe',
 'crh_Latn': 'Tatar',
 'dik_Latn': 'Dinka',
 'dyu_Latn': 'Dyula',
 'dzo_Tibt': 'Dzongkha',
 'epo_Latn': 'Esperanto',
 'eus_Latn': 'Basque',
 'ewe_Latn': 'Ewe',
 'fao_Latn': 'Faroese',
 'fij_Latn': 'Fijian',
 'fon_Latn': 'Fon',
 'fur_Latn': 'Friulian',
 'fuv_Latn': 'Fulfulde',
 'gla_Latn': 'Gaelic',
 'grn_Latn': 'Guarani',
 'hat_Latn': 'Creole',
 'hne_Deva': 'Chhattisgarhi',
 'ilo_Latn': 'Ilocano',
 'kab_Latn': 'Kabyle',
 'kac_Latn': 'Jingpho',
 'kas_Arab': 'Kashmiri',
 'kas_Deva': 'Kashmiri',
 'kbp_Latn': 'Kabiyè',
 'kik_Latn': 'Kikuyu',
 'kin_Latn': 'Kinyarwanda',
 'kir_Cyrl': 'Kyrgyz',
 'kmb_Latn': 'Kimbundu',
 'kmr_Latn': 'Kurdish',
 'knc_Arab': 'Kanuri',
 'knc_Latn': 'Kanuri',
 'kon_Latn': 'Kikongo',
 'lij_Latn': 'Ligurian',
 'lim_Latn': 'Limburgish',
 'lmo_Latn': 'Lombard',
 'ltg_Latn': 'Latgalian',
 'lua_Latn': 'Luba-Kasai',
 'lus_Latn': 'Mizo',
 'mag_Deva': 'Magahi',
 'mai_Deva': 'Maithili',
 'min_Arab': 'Minangkabau',
 'min_Latn': 'Minangkabau',
 'mni_Beng': 'Meitei',
 'mos_Latn': 'Mossi',
 'nno_Latn': 'Norwegian',
 'nus_Latn': 'Nuer',
 'pag_Latn': 'Pangasinan',
 'pap_Latn': 'Papiamento',
 'plt_Latn': 'Malagasy',
 'prs_Arab': 'Dari',
 'quy_Latn': 'Quechua',
 'run_Latn': 'Rundi',
 'sag_Latn': 'Sango',
 'san_Deva': 'Sanskrit',
 'sat_Olck': 'Santali',
 'scn_Latn': 'Sicilian',
 'shn_Mymr': 'Shan',
 'sin_Sinh': 'Sinhala',
 'sot_Latn': 'Sotho',
 'srd_Latn': 'Sardinian',
 'ssw_Latn': 'Swati',
 'sun_Latn': 'Sundanese',
 'szl_Latn': 'Silesian',
 'taq_Latn': 'Tamasheq',
 'taq_Tfng': 'Tamasheq',
 'tat_Cyrl': 'Tatar',
 'tir_Ethi': 'Tigrinya',
 'tpi_Latn': 'Pisin',
 'tsn_Latn': 'Tswana',
 'tso_Latn': 'Tsonga',
 'tuk_Latn': 'Turkmen',
 'tum_Latn': 'Tumbuka',
 'twi_Latn': 'Twi',
 'tzm_Tfng': 'Tamazight',
 'uig_Arab': 'Uyghur',
 'vec_Latn': 'Venetian',
 'war_Latn': 'Waray',
 'ydd_Hebr': 'Yiddish',
 'yue_Hant': 'Chinese'}

combined = {**flores_l, **remaining_l}

languages = list(combined.keys())
languages_to_run = languages


prompt_lid= "You are an assistant specialized in identifying written languages. Please identify the language in the text below. Return only the name of the language, no other text.\n Text: {}"

import time

import requests
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

def get_gemma3_reply(prompt):
    data = {
        "model": "google/gemma-3n-E4B-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    completion = response.json()

    content = completion["choices"][0]["message"]["content"]
    return content

def get_category(prompt, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            response = get_gemma3_reply(prompt)
            if response is not None:
                return ' '.join(response.split())
            else:
                return "invalid"
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Returning failure label.")
                return "invalid"  
                
from tqdm import tqdm
result_categories = {}
result_accuracies = {}
gpt_replies = {}

for language_code in languages_to_run:
    completed = []
    if language_code in completed:
        continue
    print(language_code)
    accurate = 0
    
    result_categories[language_code]=[]
    gpt_replies[language_code] = []

    language_name = combined[language_code]
    
    size = 1012
    for i in tqdm(range(size)): #length of devtest
        sentence = dataset['devtest'][i]["sentence_{}".format(language_code)]
        prompt = prompt_lid.format(sentence)
        # print(prompt)
        reply = get_category(prompt)
        gpt_replies[language_code].append(reply)

        category = ""
        if language_name.lower() in reply.lower():
            category = language_name
        result_categories[language_code].append(category)
        if category == language_name:
            accurate+=1
            
    result_accuracies[language_code] = accurate

    df = pd.DataFrame({
    "answer": result_categories[language_code]
    })

    # Save to CSV
    df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)


    df2 = pd.DataFrame({
    "reply": gpt_replies[language_code]
    })

    # Save to CSV
    df2.to_csv("{}/{}.csv".format(results_reply_folder,language_code), index=False)
    print(accurate)
print(result_accuracies)

