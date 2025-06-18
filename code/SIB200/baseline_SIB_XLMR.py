from transformers import AutoTokenizer
from datasets import load_dataset

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

from transformers import AutoModelForSequenceClassification

from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


import argparse

# results_folder = '../results/SIB200/sib_results_XLMR'
# results_csv_folder = '../csvs/SIB200'

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True)
parser.add_argument("--results_csv_folder", type=str, required=True)

args = parser.parse_args()

results_folder = args.results_folder
results_csv_folder = args.results_csv_folder


sib_languages = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'als_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'arb_Latn', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gaz_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khk_Cyrl', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kmr_Latn', 'knc_Arab', 'knc_Latn', 'kon_Latn', 'kor_Hang', 'lao_Laoo', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'lvs_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Arab', 'min_Latn', 'mkd_Cyrl', 'mlt_Latn', 'mni_Beng', 'mos_Latn', 'mri_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nqo_Nkoo', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pbt_Arab', 'pes_Arab', 'plt_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Olck', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'taq_Latn', 'taq_Tfng', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zsm_Latn', 'zul_Latn']
languages_to_run = sib_languages


categories = ['science/technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography']

# Define label2id and id2label maps
label2id = {label: i for i, label in enumerate(categories)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example["label"] = label2id[example["category"]]
    return example


# This is to delete checkpoints when better checkpoint is found. Space constraints. 

from transformers import TrainerCallback, TrainingArguments, Trainer
import os
import shutil

class CleanCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.best_checkpoint = None

    def on_save(self, args, state, control, **kwargs):
        # Keep only the newest checkpoint
        current_ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        
        # Delete the previous best if it exists and is different
        if self.best_checkpoint and self.best_checkpoint != current_ckpt_dir:
            try:
                shutil.rmtree(self.best_checkpoint)
                print(f"Deleted old checkpoint: {self.best_checkpoint}")
            except Exception as e:
                print(f"Could not delete {self.best_checkpoint}: {e}")

        # Update the latest/best checkpoint
        self.best_checkpoint = current_ckpt_dir

from tqdm import tqdm
result_categories = {}
result_accuracies = {}

for language_code in languages_to_run:
    completed = []
    if language_code in completed:
        continue
    print(language_code)
    accurate = 0

    result_categories[language_code]=[]
    dataset = load_dataset("Davlan/sib200", language_code) 
    dataset = dataset.map(encode_labels) #change text to number labels. 
    size = len(dataset['test'])

    encoded_dataset = dataset.map(preprocess, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CleanCheckpointCallback(training_args.output_dir)]
    )
    trainer.train()

    # Predict on test
    preds_output = trainer.predict(encoded_dataset["test"])
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    # Accuracy
    acc = (preds == labels).mean()
    result_accuracies[language_code] = acc    
    # Decode predictions
    pred_labels = [id2label[p] for p in preds]
    result_categories[language_code] = pred_labels
    
    df = pd.DataFrame({
    "text": dataset['test']['text'],
    "gpt_label": result_categories[language_code]
    })

    # Save to CSV
    df.to_csv("{}/{}.csv".format(results_folder,language_code), index=False)
    print(acc)

    import shutil
    import os
    
    # Delete all intermediate checkpoints
    checkpoint_dir = "./results"
    for subdir in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, subdir)
        if subdir.startswith("checkpoint"):
            shutil.rmtree(full_path)

print(result_accuracies)

import pandas as pd
df = pd.read_csv('csvs/SIB200/sib_gpt4o_results_april2025.csv')
code_to_name_map = df.set_index('language code')['Language name'].to_dict()
df = pd.DataFrame({
    "language_name": [code_to_name_map[language_code] for language_code in sib_languages],
    "language_code": sib_languages,
    "Accuracy": [round(result_accuracies[language_code]*100, 1) for language_code in sib_languages]
})

# Save to CSV
df.to_csv("{}/{}.csv".format(results_csv_folder,"sib_xlmr_results.csv"), index=False)