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
import torch
from transformers import BitsAndBytesConfig
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# Load MNLI dataset
mnli = load_dataset("nyu-mll/multi_nli")
#mnli["train"] = mnli["train"].select(range(100))
#mnli["validation_matched"] = mnli["validation_matched"].select(range(100))
model_name = "fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,

    num_labels=3, trust_remote_code=True).to(device)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout = 0.05,
    target_modules=r".*llm2vec.*(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj).*",
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)


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
    if isinstance(logits, tuple):  # handle case where logits is a tuple
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


from math import ceil

# Dataset size and batch size
train_batch_size = 8
num_train_samples = len(encoded["train"])
steps_per_epoch = ceil(num_train_samples / train_batch_size)
eval_steps = steps_per_epoch // 5  # 20% of an epoch
# Training arguments
training_args = TrainingArguments(
    output_dir="./nllb-mnli",
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_steps = eval_steps,
    save_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # if using GPU with mixed precision
    report_to="none",  # disable wandb
    load_best_model_at_end=True,
    logging_steps=eval_steps,
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
model.save_pretrained("nllbllm2vec-mnli-finetuned")
tokenizer.save_pretrained("nllbllm2vec-mnli-finetuned")


import pandas as pd
import numpy as np
from datasets import Dataset

parent_folder = '../../GlobalNLI_dataset'
language_codes = [name for name in os.listdir(parent_folder)
                if os.path.isdir(os.path.join(parent_folder, name))]
language_codes.sort()
languages_to_run = language_codes

from tqdm import tqdm
# labels = {}
result_accuracies = {}

for language_code in languages_to_run:
    print(language_code)
    # labels[language_code]=[]
    result_accuracies[language_code] = []

    # Load and prepare test dataset
    df = pd.read_csv("../../GlobalNLI_dataset/{}/test.csv".format(language_code))
    test_dataset = Dataset.from_pandas(df)

    # Preprocess
    test_dataset = test_dataset.map(preprocess, batched=True)
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Run evaluation
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions[0], axis=-1)
    df["predicted_label"] = preds
    df["correct"] = (df["label"] == df["predicted_label"]).astype(int)

    # Compute accuracy
    accurate = df["correct"].sum()


    result_accuracies[language_code] = accurate

    result_df = pd.DataFrame({
    "premise": df['premise'],
    "hypothesis": df['hypothesis'],
    "gpt_label": preds
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

import os

parent_folder = 'nli_dataset'
language_codes = [name for name in os.listdir(parent_folder)
                if os.path.isdir(os.path.join(parent_folder, name))]
language_codes.sort()
import pandas as pd
df = pd.DataFrame({
    "Language name": [lang_code_to_name[language_code] for language_code in language_codes],
    "Language code": language_codes,
    "Accuracy": [round(result_accuracies[language_code]*100/600, 1) for language_code in language_codes]
})

# Save to CSV
df.to_csv("{}/{}.csv".format(results_csv_folder,'nli_results_nllbllm2vec.csv'), index=False)
