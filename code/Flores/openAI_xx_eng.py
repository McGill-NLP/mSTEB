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

from openai import OpenAI
client = OpenAI()

# Load the FLORES dataset from Hugging Face
dataset = load_dataset("facebook/flores", "all")  # Loads all languages

language_mapping = pd.read_csv('../csvs/language_mapping.csv')
language_mapping_index = language_mapping.set_index('FLORES-200 code')['Language'].to_dict()
all_languages = list(language_mapping_index.keys())
all_languages.remove('eng_Latn')

prompt_xx_eng = "You are a translation expert. Translate the following {} sentences to English \n{} sentence: {}\nEnglish sentence: . Return only the translated sentence."

import re
def get_translation(prompt, errors):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ])
    reply = completion.choices[0].message.content
    reply = ' '.join(reply.split()) #convert it to one line

    # print(reply)
    return reply

languages_to_run = all_languages[0:30]
completed = []
from tqdm import tqdm
translations = {}

for language_name in languages_to_run:
    if language_name in completed:
        continue
    print(language_name)
    errors = 0
    translations[language_name]=[]
    for i in tqdm(range(1012)): #length of devtest
        sentence_xx = dataset['devtest'][i]["sentence_{}".format(language_name)]
        prompt = prompt_xx_eng.format(language_mapping_index[language_name],language_mapping_index[language_name],sentence_xx)
        english_translation = get_translation(prompt, errors)
        translations[language_name].append(english_translation)
        
    with open("{}/{}.txt".format(results_translations_folder_xx_eng,language_name), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in translations[language_name])