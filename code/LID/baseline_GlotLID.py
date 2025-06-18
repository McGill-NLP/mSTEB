# pip install fasttext
# pip install huggingface_hub

import fasttext
from huggingface_hub import hf_hub_download

# download model and get the model path
# cache_dir is the path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)
print("model path:", model_path)

# load the model
model = fasttext.load_model(model_path)

import argparse

# results_folder = '../results/LID/lid_predicted_GlotLID'
# results_csv_folder = '../csvs/LID'

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True)
parser.add_argument("--results_csv_folder", type=str, required=True)

args = parser.parse_args()

results_folder = args.results_folder
results_csv_folder = args.results_csv_folder


import pandas as pd
language_mapping = pd.read_csv('csvs/language_mapping.csv')
language_mapping_index = language_mapping.set_index('FLORES-200 code')['Language'].to_dict()
all_languages = list(language_mapping_index.keys())
from datasets import load_dataset
dataset = load_dataset("facebook/flores", "all")  # Loads all languages

non_existant = []
for lang in all_languages:
    if "__label__{}".format(lang) not in model.labels:
        non_existant.append(lang)

print(non_existant)

languages_to_run = all_languages

from tqdm import tqdm
result_categories = {}
result_percentages = {}
result_accuracies = {}

for language_code in languages_to_run:
    completed = []
    if language_code in completed:
        continue
    print(language_code)
    accurate = 0
    
    result_categories[language_code]=[]
    result_percentages[language_code]=[]
    
    size = 1012
    for i in tqdm(range(size)): #length of devtest
        sentence = dataset['devtest'][i]["sentence_{}".format(language_code)]
        reply = model.predict(sentence)
        pred_lang = reply[0][0][9:]
        pred_percent = round(reply[1][0],2)
        result_categories[language_code].append(pred_lang)
        result_percentages[language_code].append(pred_percent)
        
        if pred_lang == language_code:
            accurate+=1
            
    result_accuracies[language_code] = accurate

    df = pd.DataFrame({
    "answer": result_categories[language_code],
    "percentage": result_percentages[language_code]
    })

    # Save to CSV
    df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)

    print(accurate)
print(result_accuracies)

redo_l={'aka_Latn':'twi_Latn','est_Latn':'ekk_Latn','grn_Latn':'gug_Latn','kon_Latn':'kng_Latn','tgl_Latn':'fil_Latn','pes_Arab':'fas_Arab','prs_Arab':'fas_Arab','zho_Hans':'cmn_Hani'}
import os
import pandas as pd
for language_code in list(redo_l.keys()):
    correct_answer = redo_l[language_code]
    result_list = []
    
    reply_path = os.path.join(results_folder, "{}.csv".format(language_code))
    df = pd.read_csv(reply_path)
    acc = 0 
    for i in range(1012):
        reply = df['answer'][i]
        if reply == correct_answer:
            acc+=1
            
    result_accuracies[language_code]=acc
    print(acc)

