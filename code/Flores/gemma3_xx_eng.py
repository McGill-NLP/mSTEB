# results_translations_folder_xx_eng = '../results/Flores/translations_gemma3_xx_eng'
# results_translations_folder_eng_xx = '../results/Flores/translations_gemma3_eng_xx'

# reference_xx_eng_folder = '../results/Flores/reference_xx_eng'
# reference_eng_xx_folder = '../results/Flores/reference_eng_xx'
# results_csv_folder = '../csvs/Flores'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_translations_folder_xx_eng", type=str, required=True)

args = parser.parse_args()

results_translations_folder_xx_eng = args.results_translations_folder_xx_eng

import os
import openai
import json
import pandas as pd
from datasets import load_dataset

# Load the FLORES dataset from Hugging Face
dataset = load_dataset("facebook/flores", "all")  # Loads all languages

language_mapping = pd.read_csv('../csvs/language_mapping.csv')
language_mapping_index = language_mapping.set_index('FLORES-200 code')['Language'].to_dict()
all_languages = list(language_mapping_index.keys())
all_languages.remove('eng_Latn')

prompt_xx_eng = "You are a translation expert. Translate the following {} sentences to English \n{} sentence: {}\nEnglish sentence: . Return only the translated sentence."

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
    
def get_translation(prompt, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            reply = get_gemma3_reply(prompt)
            # print(reply)
            reply = ' '.join(reply.split()) #convert it to one line
            return reply
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Returning failure label.")
                return ""   



languages_to_run = all_languages
completed = []

from tqdm import tqdm
translations = {}
result_inaccuracies = {}

for language_name in languages_to_run:
    if language_name in completed:
        continue
    print(language_name)
    errors = 0
    translations[language_name]=[]
    for i in tqdm(range(1012)): #length of devtest
        sentence_xx = dataset['devtest'][i]["sentence_{}".format(language_name)]
        prompt = prompt_xx_eng.format(language_mapping_index[language_name],language_mapping_index[language_name],sentence_xx)
        english_translation = get_translation(prompt)
        translations[language_name].append(english_translation)

        if english_translation=="":
            errors+=1
    
    result_inaccuracies[language_name]=errors
    with open("{}/{}.txt".format(results_translations_folder_xx_eng,language_name), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in translations[language_name])
