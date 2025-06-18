import os
import openai
import json
import pandas as pd
from datasets import load_dataset

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

import argparse

# results_folder = '../results/Belebele/belebele_results_gemini2'
# results_reply_folder = '../results/Belebele/belebele_replies_gemini2'

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True)
parser.add_argument("--results_reply_folder", type=str, required=True)

args = parser.parse_args()

results_folder = args.results_folder
results_reply_folder = args.results_reply_folder



languages = [
    "acm_Arab", "afr_Latn", "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "arb_Latn", "ars_Arab",
    "ary_Arab", "arz_Arab", "asm_Beng", "azj_Latn", "bam_Latn", "ben_Beng", "ben_Latn", "bod_Tibt",
    "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "ckb_Arab", "dan_Latn", "deu_Latn", "ell_Grek",
    "eng_Latn", "est_Latn", "eus_Latn", "fin_Latn", "fra_Latn", "fuv_Latn", "gaz_Latn", "grn_Latn",
    "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hin_Latn", "hrv_Latn", "hun_Latn",
    "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan",
    "kac_Latn", "kan_Knda", "kat_Geor", "kaz_Cyrl", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kin_Latn",
    "kir_Cyrl", "kor_Hang", "lao_Laoo", "lin_Latn", "lit_Latn", "lug_Latn", "luo_Latn", "lvs_Latn",
    "mal_Mlym", "mar_Deva", "mkd_Cyrl", "mlt_Latn", "mri_Latn", "mya_Mymr", "nld_Latn", "nob_Latn",
    "npi_Deva", "npi_Latn", "nso_Latn", "nya_Latn", "ory_Orya", "pan_Guru", "pbt_Arab", "pes_Arab",
    "plt_Latn", "pol_Latn", "por_Latn", "ron_Latn", "rus_Cyrl", "shn_Mymr", "sin_Latn", "sin_Sinh",
    "slk_Latn", "slv_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "srp_Cyrl",
    "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "tam_Taml", "tel_Telu", "tgk_Cyrl", "tgl_Latn",
    "tha_Thai", "tir_Ethi", "tsn_Latn", "tso_Latn", "tur_Latn", "ukr_Cyrl", "urd_Arab", "urd_Latn",
    "uzn_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "yor_Latn", "zho_Hans", "zho_Hant",
    "zsm_Latn", "zul_Latn"
]

languages_to_run = languages[0:122]

prompt_belebele= "P: {}\nQ: {}\nA: {}\nB: {}\nC: {}\nD: {}\nPlease choose the correct answer from the options above:"

#this logic is taken from afrobench. 
import re

# Hardcoded choices
choices = ["A", "B", "C", "D"]

# Hardcoded verbalizer mapping
verbalizer = {
    "A": ['a:', 'a', 'a.', '1', '1:', 'a)', '(a)', 'option a', 'option a:', 'option_a:'],
    "B": ['b:', 'b', 'b.', '2', '2:', 'b)', '(b)', 'option b', 'option b:', 'option_b:'],
    "C": ['c:', 'c', 'c.', '3', '3:', 'c)', '(c)', 'option c', 'option c:', 'option_c:'],
    "D": ['d:', 'd', 'd.', '4', '4:', 'd)', '(d)', 'option d', 'option d:', 'option_d:'],
}

map_response = {
    "A": '1',
    "B": '2',
    "C": '3',
    "D": '4',
    "invalid": "invalid"
}

import time

def get_category(prompt, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            response =response.text
            # print(response)
            choice_patterns = {choice: re.compile(rf"\b{re.escape(choice)}\b", re.IGNORECASE) for choice in choices}
        
            best_match = None
            best_position = len(response) + 1         
            # Here check for all 4, and choose the best position. 
            for choice, pattern in choice_patterns.items():
                match = pattern.search(response)
                if match and match.start() < best_position:
                    best_match = choice
                    best_position = match.start()
        
            # Check against verbalizer if no match found, and take the first hit
            if not best_match and verbalizer:
                for key, synonyms in verbalizer.items():
                    for synonym in synonyms:
                        # Use \b for word boundaries to avoid matching parts of words
                        if re.search(rf"\b{re.escape(synonym)}\b", response.lower(), re.IGNORECASE):
                            best_match = key
                            break
                    if best_match:
                        break
            # Append result
            if best_match:
                answer = best_match
            else:
                answer = "invalid"
                
            return response, map_response[answer]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Returning failure label.")
                return "","invalid"   
                


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
    dataset = load_dataset("facebook/belebele", language_code)

    size = len(dataset['test'])
    for i in tqdm(range(size)): #length of devtest
        flores_passage = dataset['test'][i]['flores_passage']
        question = dataset['test'][i]['question']
        mc_answer1 = dataset['test'][i]['mc_answer1']
        mc_answer2 = dataset['test'][i]['mc_answer2']
        mc_answer3 = dataset['test'][i]['mc_answer3']
        mc_answer4 = dataset['test'][i]['mc_answer4']
        
        prompt = prompt_belebele.format(flores_passage, question, mc_answer1, mc_answer2, mc_answer3, mc_answer4)
        # print(prompt)
        reply, category = get_category(prompt)
        gpt_replies[language_code].append(reply)
            
        result_categories[language_code].append(category)
        if category == dataset['test'][i]['correct_answer_num']:
            accurate+=1
    result_accuracies[language_code] = accurate

    df = pd.DataFrame({
    "question": dataset['test']['question'],
    "gemini_answer": result_categories[language_code]
    })

    # Save to CSV
    df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)

    df2 = pd.DataFrame({
    "question": dataset['test']['question'],
    "gemini_reply": gpt_replies[language_code]
    })

    # Save to CSV
    df2.to_csv("{}/{}.csv".format(results_reply_folder,language_code), index=False)
    print(accurate)

print(result_accuracies)
