import argparse

# results_folder = '../results/GLobalNLI/nli_predicted_labels_mdeberta'
# results_csv_folder = '../csvs/GLobalNLI'

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True)
parser.add_argument("--results_csv_folder", type=str, required=True)

args = parser.parse_args()

results_folder = args.results_folder
results_csv_folder = args.results_csv_folder


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np

# Load MNLI dataset
mnli = load_dataset("nyu-mll/multi_nli")

# Load tokenizer and model
model_name = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Preprocess
def preprocess(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

encoded = mnli.map(preprocess, batched=True)
encoded = encoded.rename_column("label", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Evaluation metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)
# Training arguments
training_args = TrainingArguments(
    output_dir="./mdeberta-mnli",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # if using GPU with mixed precision
    report_to="none",  # disable wandb
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation_matched"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

model.save_pretrained("mdeberta-v3-mnli-finetuned")
tokenizer.save_pretrained("mdeberta-v3-mnli-finetuned")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("mdeberta-v3-mnli-finetuned")
tokenizer = AutoTokenizer.from_pretrained("mdeberta-v3-mnli-finetuned")
model.eval()


import os
import pandas as pd

parent_folder = '../../GlobalNLI_dataset'
language_codes = [name for name in os.listdir(parent_folder)
                if os.path.isdir(os.path.join(parent_folder, name))]
language_codes.sort()
languages_to_run = language_codes

from tqdm import tqdm
labels = {}
result_accuracies = {}

for language_code in languages_to_run:
    print(language_code)
    labels[language_code]=[]
    result_accuracies[language_code] = []
    accurate = 0
    
    df = pd.read_csv("../../GlobalNLI_dataset/{}/test.csv".format(language_code))
    
    for i in tqdm(range(600)): #length of devtest
        premise = df.iloc[i]['premise']
        hypothesis = df.iloc[i]['hypothesis']
        gold_label = df.iloc[i]['label']
        
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            
        labels[language_code].append(pred)
        if pred == gold_label:
            accurate+=1
            
    result_accuracies[language_code] = accurate
    
    result_df = pd.DataFrame({
    "premise": df['premise'],
    "hypothesis": df['hypothesis'],
    "gpt_label": labels[language_code]
    })

    # Save to CSV
    result_df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)

    print(accurate)

print(result_accuracies)

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
import pandas as pd
df = pd.DataFrame({
    "Language name": [lang_code_to_name[language_code] for language_code in language_codes],
    "Language code": language_codes,
    "Accuracy": [round(result_accuracies[language_code]*100/600, 1) for language_code in language_codes]
})

# Save to CSV
df.to_csv("{}/{}.csv".format(results_csv_folder,'nli_results_mDeBERTa.csv'), index=False)
