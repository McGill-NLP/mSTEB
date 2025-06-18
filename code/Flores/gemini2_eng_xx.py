# results_translations_folder_xx_eng = '../results/Flores/translations_gemini_2_xx_eng'
# results_translations_folder_eng_xx = '../results/Flores/translations_gemini_2_eng_xx'

# reference_xx_eng_folder = '../results/Flores/reference_xx_eng'
# reference_eng_xx_folder = '../results/Flores/reference_eng_xx'
# results_csv_folder = '../csvs/Flores'
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

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))



# Load the FLORES dataset from Hugging Face
dataset = load_dataset("facebook/flores", "all")  # Loads all languages

language_mapping = pd.read_csv('../csvs/language_mapping.csv')
language_mapping_index = language_mapping.set_index('FLORES-200 code')['Language'].to_dict()
all_languages = list(language_mapping_index.keys())
all_languages.remove('eng_Latn')

prompt_eng_xx = "As a English and {} linguist, translate the following English sentences to {}. \nEnglish sentence: {}\n{} sentence: . Return only the translated sentence."

import time

def get_translation(prompt, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            reply =response.text
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

        english_sentence = dataset['devtest'][i]["sentence_eng_Latn"]
        prompt = prompt_eng_xx.format(language_mapping_index[language_name],language_mapping_index[language_name],english_sentence,language_mapping_index[language_name])
        translated_sentence = get_translation(prompt)
        translations[language_name].append(translated_sentence)
        if translated_sentence=="":
            errors+=1  
    result_inaccuracies[language_name]=errors

    with open("{}/{}.txt".format(results_translations_folder_eng_xx,language_name), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in translations[language_name])

