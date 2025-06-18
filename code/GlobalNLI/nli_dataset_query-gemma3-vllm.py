import argparse

# results_folder = '../results/GLobalNLI/nli_predicted_labels_gemma3'
# results_reply_folder = '../results/GLobalNLI/nli_replies_gemma3'
# results_csv_folder = '../csvs/GLobalNLI'

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True)
parser.add_argument("--results_reply_folder", type=str, required=True)
parser.add_argument("--results_csv_folder", type=str, required=True)

args = parser.parse_args()

results_folder = args.results_folder
results_reply_folder = args.results_reply_folder
results_csv_folder = args.results_csv_folder


import os
import json
import pandas as pd
from datasets import load_dataset

import requests


import os
parent_folder = '../../GlobalNLI_dataset'
language_codes = [name for name in os.listdir(parent_folder)
                if os.path.isdir(os.path.join(parent_folder, name))]
language_codes.sort()


lang_code_to_name = {
    'amh': 'Amharic',
    'ara': 'Arabic',
    'asm': 'Assamese',
    'aym': 'Aymara',
    'ben': 'Bengali',
    'bul': 'Bulgarian',
    'bzd': 'Bribri',
    'cat': 'Catalan',
    'cni': 'Asháninka',
    'deu': 'German',
    'ell': 'Greek',
    'eng': 'English',
    'ewe': 'Ewe',
    'fra': 'French',
    'grn': 'Guarani',
    'guj': 'Gujarati',
    'hau': 'Hausa',
    'hch': 'Wixarika',
    'hin': 'Hindi',
    'ibo': 'Igbo',
    'ind': 'Indonesian',
    'jpn': 'Japanese',
    'kan': 'Kannada',
    'kin': 'Kinyarwanda',
    'kor': 'Korean',
    'lin': 'Lingala',
    'lug': 'Luganda',
    'mal': 'Malayalam',
    'mar': 'Marathi',
    'mya': 'Burmese',
    'nah': 'Nahuatl',
    'ori': 'Odia (Oriya)',
    'orm': 'Oromo',
    'oto': 'Otomi',
    'pan': 'Punjabi',
    'pat': 'Jamaican Patois',
    'pol': 'Polish',
    'por': 'Portuguese',
    'quy': 'Quechua',
    'ron': 'Romanian',
    'rus': 'Russian',
    'shp': 'Shipibo-Conibo',
    'sna': 'chiShona',
    'sot': 'Sesotho',
    'spa': 'Spanish',
    'swa': 'Swahili',
    'tam': 'Tamil',
    'tar': 'Rarámuri',
    'tel': 'Telugu',
    'tha': 'Thai',
    'tur': 'Turkish',
    'twi': 'Twi',
    'urd': 'Urdu',
    'vie': 'Vietnamese',
    'wol': 'Wolof',
    'xho': 'isiXhosa',
    'yor': 'Yoruba',
    'zho': 'Chinese',
    'zul': 'isiZulu'
}
# "Given the following premise and hypothesis in {{language}}, identify if the premise entails, contradicts, or is neutral towards the hypothesis. Please respond with exact 'entailment', 'contradiction', or 'neutral'. \n\nPremise: {{premise}} \nHypothesis: {{hypothesis}}"
prompt_nli = "Given the following premise and hypothesis in {}, identify if the premise entails, contradicts, or is neutral towards the hypothesis. Please respond with exact 'entailment', 'contradiction', or 'neutral'. \n\nPremise: {} \nHypothesis: {}"

import time
import re

import requests
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

def get_gemma3_reply(prompt):
    data = {
        "model": "google/gemma-3-27b-it",
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

def get_label(prompt, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            reply = get_gemma3_reply(prompt)
            reply = reply.lower()
            # print(reply)
            if 'entail' in reply:
                label = 0
            elif 'neutral' in reply:
                label = 1
            elif 'contradict' in reply:
                label = 2
            else:
                label = -1

            return reply, label

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Returning failure label.")
                return "", -1


languages_to_run = language_codes

from tqdm import tqdm
labels = {}
gpt_replies = {}
result_accuracies = {}
result_inaccuracies = {}

for language_code in languages_to_run:
    # if language_code == 'eng':
    #     continue
    print(language_code)
    labels[language_code]=[]
    gpt_replies[language_code]=[]
    result_accuracies[language_code] = 0
    result_inaccuracies[language_code] = 0

    accurate = 0
    errors = 0
    df = pd.read_csv("../../GlobalNLI_dataset/{}/test.csv".format(language_code))
    
    for i in tqdm(range(600)): #length of devtest
        premise = df.iloc[i]['premise']
        hypothesis = df.iloc[i]['hypothesis']
        gold_label = df.iloc[i]['label']
        
        prompt = prompt_nli.format(lang_code_to_name[language_code],premise,hypothesis)
        gpt_reply, label = get_label(prompt)
        gpt_replies[language_code].append(gpt_reply)
        labels[language_code].append(label)
        if label == gold_label:
            accurate+=1
        if label == -1:
            errors +=1
            
    result_accuracies[language_code] = accurate
    result_inaccuracies[language_code] = errors

    result_df = pd.DataFrame({
    "premise": df['premise'],
    "hypothesis": df['hypothesis'],
    "gpt_label": labels[language_code]
    })

    # Save to CSV
    result_df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)

    df_gpt_replies = pd.DataFrame({
        "premise": df['premise'],
        "hypothesis": df['hypothesis'],
        "gpt_reply": gpt_replies[language_code]
    })

    df_gpt_replies.to_csv("{}/{}.csv".format(results_reply_folder,language_code), index=False)

import json

# Save to a JSON file
with open("result_accuracies_nli_gemma3.json", "w", encoding="utf-8") as f:
    json.dump(result_accuracies, f, ensure_ascii=False, indent=4)
    
with open("result_inaccuracies_nli_gemma3.json", "w", encoding="utf-8") as f:
    json.dump(result_inaccuracies, f, ensure_ascii=False, indent=4)

df = pd.DataFrame({
    "Language name": [lang_code_to_name[language_code] for language_code in language_codes],
    "Language code": language_codes,
    "Accuracy": [round(result_accuracies[language_code]*100/600, 1) for language_code in language_codes]
})

# Save to CSV
df.to_csv("{}/{}.csv".format(results_csv_folder,'nli_results_gemma3.csv'), index=False)

dataset_to_lang_codes = {
    'XNLI': [
        'eng', 'fra', 'spa', 'deu', 'ell', 'bul', 'rus', 'tur',
        'ara', 'vie', 'tha', 'zho', 'hin', 'swa', 'urd'
    ],
    'AfriXNLI': [
        # West Africa
        'ewe', 'hau', 'ibo', 'twi', 'wol', 'yor',
        # East Africa
        'amh', 'kin', 'lug', 'swa', 'orm',
        # Southern Africa
        'sna', 'xho', 'zul', 'sot',
        # Central Africa
        'lin'
    ],
    'IndicXNLI': [
        'asm', 'guj', 'kan', 'mal', 'mar', 'ori', 'pan',
        'tam', 'tel', 'hin', 'ben'
    ],
    'AmericasXNLI': [
        'aym', 'cni', 'bzd', 'grn', 'nah', 'oto', 'quy', 'tar', 'shp', 'hch'
    ],
    'XNLI-ca': ['cat'],
    'myXNLI': ['mya'],
    'IndoNLI': ['ind'],
    'JNLI': ['jpn'],
    'Portugese': ['por'],
    'Polish': ['pol'],
    'JamPatoisNLI': ['pat'],
    'Korean': ['kor'],
    'Romainian': ['ron']
}
#create dataset to average performance. 


import pandas as pd

# Assuming df, lang_code_to_name, result_accuracies, language_codes, and dataset_to_lang_codes are already defined

# Create a mapping from language code to accuracy for fast lookup
lang_code_to_accuracy = dict(zip(df["Language code"], df["Accuracy"]))

# Compute average accuracy per dataset
dataset_avg_accuracies = {}

for dataset, lang_codes in dataset_to_lang_codes.items():
    # Filter to only those lang codes that are in df
    valid_langs = [code for code in lang_codes if code in lang_code_to_accuracy]
    
    if valid_langs:
        scores = [lang_code_to_accuracy[code] for code in valid_langs]
        avg_accuracy = round(sum(scores) / len(scores), 1)
        dataset_avg_accuracies[dataset] = avg_accuracy

# Convert to a DataFrame if desired
dataset_avg_df = pd.DataFrame.from_dict(dataset_avg_accuracies, orient='index', columns=['Average Accuracy']).reset_index()
dataset_avg_df = dataset_avg_df.rename(columns={'index': 'Dataset'})

print(dataset_avg_df)


dataset_avg_df.to_csv("{}/{}.csv".format(results_csv_folder,'nli_results_gemma3_april2025_by_family.csv'), index=False)
