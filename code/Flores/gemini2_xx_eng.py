# results_translations_folder_xx_eng = '../results/Flores/translations_gemini_2_xx_eng'
# results_translations_folder_eng_xx = '../results/Flores/translations_gemini_2_eng_xx'

# reference_xx_eng_folder = '../results/Flores/reference_xx_eng'
# reference_eng_xx_folder = '../results/Flores/reference_eng_xx'
# results_csv_folder = '../csvs/Flores'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_translations_folder_xx_eng", type=str, required=True)
parser.add_argument("--results_translations_folder_eng_xx", type=str, required=True)
parser.add_argument("--reference_xx_eng_folder", type=str, required=True)
parser.add_argument("--reference_eng_xx_folder", type=str, required=True)
parser.add_argument("--results_csv_folder", type=str, required=True)

args = parser.parse_args()

results_translations_folder_xx_eng = args.results_translations_folder_xx_eng
results_translations_folder_eng_xx = args.results_translations_folder_eng_xx
reference_xx_eng_folder = args.reference_xx_eng_folder
reference_eng_xx_folder = args.reference_eng_xx_folder
results_csv_folder = args.results_csv_folder


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

prompt_xx_eng = "You are a translation expert. Translate the following {} sentences to English \n{} sentence: {}\nEnglish sentence: . Return only the translated sentence."

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
        prompt = prompt_xx_eng.format(language_mapping_index[language_name],language_mapping_index[language_name],sentence_xx)
        english_translation = get_translation(prompt)
        translations[language_name].append(english_translation)

        if english_translation=="":
            errors+=1
    
    result_inaccuracies[language_name]=errors
    with open("{}/{}.txt".format(results_translations_folder_xx_eng,language_name), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in translations[language_name])



#get scores 

import os
import subprocess
import json

# Define paths
translations_dir = results_translations_folder_xx_eng
reference_file = "{}/eng_Latn.txt".format(reference_xx_eng_folder)

# Dictionary to store scores
scores = {}

# Loop over all .txt files in the translations directory
for file in os.listdir(translations_dir):
    if file.endswith(".txt"):  # Process only .txt files
        language_name = file.replace(".txt", "")  # Extract language name
        translation_file = os.path.join(translations_dir, file)  # Full path

        # Run sacrebleu command
        command = f"sacrebleu -m chrf --chrf-word-order 2 {translation_file} < {reference_file}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Extract the score from the output
        try:
            output_json = json.loads(result.stdout)  # Parse JSON output
            scores[language_name] = output_json["score"]  # Store score
        except json.JSONDecodeError:
            print(f"Error processing {language_name}: Invalid JSON output")
        except KeyError:
            print(f"Error: 'score' field not found in {language_name} output")

# Print results
print(scores)

#get scores2 for eng to xx

import os
import subprocess
import json

# Define paths
translations_dir = results_translations_folder_eng_xx
reference_dir = reference_eng_xx_folder

# Dictionary to store scores2
scores2 = {}

# Loop over all .txt files in the translations directory
for file in os.listdir(translations_dir):
    if file.endswith(".txt"):  # Process only .txt files
        language_name = file.replace(".txt", "")  # Extract language name
        translation_file = os.path.join(translations_dir, file)  # Full path
        reference_file = os.path.join(reference_dir, file)  # Full path

        # Run sacrebleu command
        command = f"sacrebleu -m chrf --chrf-word-order 2 {translation_file} < {reference_file}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Extract the score from the output
        try:
            output_json = json.loads(result.stdout)  # Parse JSON output
            scores2[language_name] = output_json["score"]  # Store score
        except json.JSONDecodeError:
            print(f"Error processing {language_name}: Invalid JSON output")
        except KeyError:
            print(f"Error: 'score' field not found in {language_name} output")

# Print results
print(scores2)

# Convert to DataFrame
all_languages = list(language_mapping_index.keys())
all_languages.remove('eng_Latn')

all_languages2 = list(language_mapping_index.values())
all_languages2.remove('English')

df = pd.DataFrame({
    "Language": all_languages2,
    "FLORES-200 code": all_languages,
    "xx - English": [scores[lang] for lang in all_languages],
    "English - xx": [scores2[lang] for lang in all_languages]
})

# Save to CSV
df.to_csv("{}/{}.csv".format(results_csv_folder,'flores_gemini2_results_april_2025.csv'), index=False)
