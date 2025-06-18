import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_translations_folder_eng_xx", type=str, required=True)

args = parser.parse_args()

results_translations_folder_eng_xx = args.results_translations_folder_eng_xx


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

# prompt_eng_xx = ("You are a translation expert. Translate the following English sentences to {}. " 
#      "English: {} "
#     "Please respond only in the format Translation:: ")

prompt_eng_xx = "As a English and {} linguist, translate the following English sentences to {}. \nEnglish sentence: {}\n{} sentence: . Return only the translated sentence."
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
    return reply

languages_to_run = all_languages
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

        english_sentence = dataset['devtest'][i]["sentence_eng_Latn"]
        prompt = prompt_eng_xx.format(language_mapping_index[language_name],language_mapping_index[language_name],english_sentence,language_mapping_index[language_name])
        translated_sentence = get_translation(prompt, errors)
        translations[language_name].append(translated_sentence)
        
    with open("{}/{}.txt".format(results_translations_folder_eng_xx,language_name), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in translations[language_name])