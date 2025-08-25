import os
import json
import pandas as pd
from datasets import load_dataset

# Belebele, Flores, LID
# results_folder_belebele = 'results/Belebele/belebele_predicted_gemma3n'
# model_name = 'gemma3n'
# results_translations_folder_xx_eng = 'results/Flores/translations_gemma3n_xx_eng'
# results_translations_folder_eng_xx = 'results/Flores/translations_gemma3n_eng_xx'
# reference_xx_eng_folder = 'results/Flores/reference_xx_eng'
# reference_eng_xx_folder = 'results/Flores/reference_eng_xx'
# reply_dir_lid = 'results/LID/lid_replies_gemma3n'
# results_dir_lid = 'results/LID/lid_predicted_gemma3n'

# python code/compiling_results/results_gemma3n.py  --results_folder_belebele='results/Belebele/belebele_predicted_gemma3n' --model_name='gemma3n' --results_translations_folder_xx_eng='results/Flores/translations_gemma3n_xx_eng' --results_translations_folder_eng_xx='results/Flores/translations_gemma3n_eng_xx' --reference_xx_eng_folder='results/Flores/reference_xx_eng' --reference_eng_xx_folder='results/Flores/reference_eng_xx' --reply_dir_lid='results/LID/lid_replies_gemma3n' --results_dir_lid='results/LID/lid_predicted_gemma3n'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder_belebele", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--results_translations_folder_xx_eng", type=str, required=True)
parser.add_argument("--results_translations_folder_eng_xx", type=str, required=True)
parser.add_argument("--reference_xx_eng_folder", type=str, required=True)
parser.add_argument("--reference_eng_xx_folder", type=str, required=True)
parser.add_argument("--reply_dir_lid", type=str, required=True)
parser.add_argument("--results_dir_lid", type=str, required=True)

args = parser.parse_args()

results_folder_belebele = args.results_folder_belebele
model_name = args.model_name
results_translations_folder_xx_eng = args.results_translations_folder_xx_eng
results_translations_folder_eng_xx = args.results_translations_folder_eng_xx
reference_xx_eng_folder = args.reference_xx_eng_folder
reference_eng_xx_folder = args.reference_eng_xx_folder
reply_dir_lid = args.reply_dir_lid
results_dir_lid = args.results_dir_lid

updated_region = {'ace_Arab': 'Asia 3',
                  'ace_Latn': 'Asia 3',
                  'acm_Arab': 'Asia 1',
                  'acq_Arab': 'Asia 1',
                  'aeb_Arab': 'Africa',
                  'afr_Latn': 'Africa',
                  'ajp_Arab': 'Asia 1',
                  'aka_Latn': 'Africa',
                  'als_Latn': 'Europe 1',
                  'amh_Ethi': 'Africa',
                  'apc_Arab': 'Asia 1',
                  'arb_Arab': 'Asia 1',
                  'arb_Latn': 'Asia 1',
                  'ars_Arab': 'Asia 1',
                  'ary_Arab': 'Africa',
                  'arz_Arab': 'Africa',
                  'asm_Beng': 'Asia 2',
                  'ast_Latn': 'Europe 1',
                  'awa_Deva': 'Asia 2',
                  'ayr_Latn': 'Americas/Oceania',
                  'azb_Arab': 'Asia 1',
                  'azj_Latn': 'Asia 1',
                  'bak_Cyrl': 'Europe 2',
                  'bam_Latn': 'Africa',
                  'ban_Latn': 'Asia 3',
                  'bel_Cyrl': 'Europe 2',
                  'bem_Latn': 'Africa',
                  'ben_Beng': 'Asia 2',
                  'bho_Deva': 'Asia 2',
                  'bjn_Arab': 'Asia 3',
                  'bjn_Latn': 'Asia 3',
                  'bod_Tibt': 'Asia 3',
                  'bos_Latn': 'Europe 1',
                  'bug_Latn': 'Asia 3',
                  'bul_Cyrl': 'Europe 2',
                  'cat_Latn': 'Europe 1',
                  'ceb_Latn': 'Asia 3',
                  'ces_Latn': 'Europe 2',
                  'cjk_Latn': 'Africa',
                  'ckb_Arab': 'Asia 1',
                  'crh_Latn': 'Europe 2',
                  'cym_Latn': 'Europe 1',
                  'dan_Latn': 'Europe 1',
                  'deu_Latn': 'Europe 1',
                  'dik_Latn': 'Africa',
                  'dyu_Latn': 'Africa',
                  'dzo_Tibt': 'Asia 2',
                  'ell_Grek': 'Europe 1',
                  'eng_Latn': 'Europe 1',
                  'epo_Latn': 'Europe 2',
                  'est_Latn': 'Europe 1',
                  'eus_Latn': 'Europe 1',
                  'ewe_Latn': 'Africa',
                  'fao_Latn': 'Europe 1',
                  'fij_Latn': 'Americas/Oceania',
                  'fin_Latn': 'Europe 1',
                  'fon_Latn': 'Africa',
                  'fra_Latn': 'Europe 1',
                  'fur_Latn': 'Europe 1',
                  'fuv_Latn': 'Africa',
                  'gaz_Latn': 'Africa',
                  'gla_Latn': 'Europe 1',
                  'gle_Latn': 'Europe 1',
                  'glg_Latn': 'Europe 1',
                  'grn_Latn': 'Americas/Oceania',
                  'guj_Gujr': 'Asia 2',
                  'hat_Latn': 'Americas/Oceania',
                  'hau_Latn': 'Africa',
                  'heb_Hebr': 'Asia 1',
                  'hin_Deva': 'Asia 2',
                  'hne_Deva': 'Asia 2',
                  'hrv_Latn': 'Europe 2',
                  'hun_Latn': 'Europe 2',
                  'hye_Armn': 'Asia 1',
                  'ibo_Latn': 'Africa',
                  'ilo_Latn': 'Asia 3',
                  'ind_Latn': 'Asia 3',
                  'isl_Latn': 'Europe 1',
                  'ita_Latn': 'Europe 1',
                  'jav_Latn': 'Asia 3',
                  'jpn_Jpan': 'Asia 4',
                  'kab_Latn': 'Africa',
                  'kac_Latn': 'Asia 3',
                  'kam_Latn': 'Africa',
                  'kan_Knda': 'Asia 2',
                  'kas_Arab': 'Asia 2',
                  'kas_Deva': 'Asia 2',
                  'kat_Geor': 'Asia 1',
                  'kaz_Cyrl': 'Asia 1',
                  'kbp_Latn': 'Africa',
                  'kea_Latn': 'Africa',
                  'khk_Cyrl': 'Asia 4',
                  'khm_Khmr': 'Asia 3',
                  'kik_Latn': 'Africa',
                  'kin_Latn': 'Africa',
                  'kir_Cyrl': 'Asia 1',
                  'kmb_Latn': 'Africa',
                  'kmr_Latn': 'Asia 1',
                  'knc_Arab': 'Africa',
                  'knc_Latn': 'Africa',
                  'kon_Latn': 'Africa',
                  'kor_Hang': 'Asia 4',
                  'lao_Laoo': 'Asia 3',
                  'lij_Latn': 'Europe 1',
                  'lim_Latn': 'Europe 1',
                  'lin_Latn': 'Africa',
                  'lit_Latn': 'Europe 1',
                  'lmo_Latn': 'Europe 1',
                  'ltg_Latn': 'Europe 2',
                  'ltz_Latn': 'Europe 1',
                  'lua_Latn': 'Africa',
                  'lug_Latn': 'Africa',
                  'luo_Latn': 'Africa',
                  'lus_Latn': 'Asia 2',
                  'lvs_Latn': 'Europe 1',
                  'mag_Deva': 'Asia 2',
                  'mai_Deva': 'Asia 2',
                  'mal_Mlym': 'Asia 2',
                  'mar_Deva': 'Asia 2',
                  'min_Arab': 'Asia 3',
                  'min_Latn': 'Asia 3',
                  'mkd_Cyrl': 'Europe 2',
                  'mlt_Latn': 'Europe 1',
                  'mni_Beng': 'Asia 2',
                  'mos_Latn': 'Africa',
                  'mri_Latn': 'Americas/Oceania',
                  'mya_Mymr': 'Asia 3',
                  'nld_Latn': 'Europe 1',
                  'nno_Latn': 'Europe 1',
                  'nob_Latn': 'Europe 1',
                  'npi_Deva': 'Asia 2',
                  'nqo_Nkoo': 'Africa',
                  'nso_Latn': 'Africa',
                  'nus_Latn': 'Africa',
                  'nya_Latn': 'Africa',
                  'oci_Latn': 'Europe 1',
                  'ory_Orya': 'Asia 2',
                  'pag_Latn': 'Asia 3',
                  'pan_Guru': 'Asia 2',
                  'pap_Latn': 'Americas/Oceania',
                  'pbt_Arab': 'Asia 1',
                  'pes_Arab': 'Asia 1',
                  'plt_Latn': 'Africa',
                  'pol_Latn': 'Europe 2',
                  'por_Latn': 'Europe 1',
                  'prs_Arab': 'Asia 1',
                  'quy_Latn': 'Americas/Oceania',
                  'ron_Latn': 'Europe 2',
                  'run_Latn': 'Africa',
                  'rus_Cyrl': 'Europe 2',
                  'sag_Latn': 'Africa',
                  'san_Deva': 'Asia 2',
                  'sat_Olck': 'Asia 2',
                  'scn_Latn': 'Europe 1',
                  'shn_Mymr': 'Asia 3',
                  'sin_Sinh': 'Asia 2',
                  'slk_Latn': 'Europe 2',
                  'slv_Latn': 'Europe 1',
                  'smo_Latn': 'Americas/Oceania',
                  'sna_Latn': 'Africa',
                  'snd_Arab': 'Asia 2',
                  'som_Latn': 'Africa',
                  'sot_Latn': 'Africa',
                  'spa_Latn': 'Europe 1',
                  'srd_Latn': 'Europe 1',
                  'srp_Cyrl': 'Europe 2',
                  'ssw_Latn': 'Africa',
                  'sun_Latn': 'Asia 3',
                  'swe_Latn': 'Europe 1',
                  'swh_Latn': 'Africa',
                  'szl_Latn': 'Europe 2',
                  'tam_Taml': 'Asia 2',
                  'taq_Latn': 'Africa',
                  'taq_Tfng': 'Africa',
                  'tat_Cyrl': 'Europe 2',
                  'tel_Telu': 'Asia 2',
                  'tgk_Cyrl': 'Asia 1',
                  'tgl_Latn': 'Asia 3',
                  'tha_Thai': 'Asia 3',
                  'tir_Ethi': 'Africa',
                  'tpi_Latn': 'Americas/Oceania',
                  'tsn_Latn': 'Africa',
                  'tso_Latn': 'Africa',
                  'tuk_Latn': 'Asia 1',
                  'tum_Latn': 'Africa',
                  'tur_Latn': 'Asia 1',
                  'twi_Latn': 'Africa',
                  'tzm_Tfng': 'Africa',
                  'uig_Arab': 'Asia 1',
                  'ukr_Cyrl': 'Europe 2',
                  'umb_Latn': 'Africa',
                  'urd_Arab': 'Asia 2',
                  'uzn_Latn': 'Asia 1',
                  'vec_Latn': 'Europe 1',
                  'vie_Latn': 'Asia 3',
                  'war_Latn': 'Asia 3',
                  'wol_Latn': 'Africa',
                  'xho_Latn': 'Africa',
                  'ydd_Hebr': 'Europe 2',
                  'yor_Latn': 'Africa',
                  'yue_Hant': 'Asia 4',
                  'zho_Hans': 'Asia 4',
                  'zho_Hant': 'Asia 4',
                  'zsm_Latn': 'Asia 3',
                  'zul_Latn': 'Africa'}

dataset = load_dataset("facebook/flores", "all")  # Loads all languages

scores_belebele = {}

for file in os.listdir(results_folder_belebele):
    if file.endswith(".csv"):  # Process only .txt files
        language_code = file.replace(".csv", "")
        dataset = load_dataset("facebook/belebele", language_code)
        correct_ans_list = dataset['test']['correct_answer_num']

        file_path = os.path.join(results_folder_belebele, file)
        df = pd.read_csv(file_path)
        predicted_ans_list = [str(x) for x in list(df.iloc[:, 1])]
        matches = sum(a == b for a, b in zip(correct_ans_list, predicted_ans_list))
        scores_belebele[language_code] = matches
print('scores_belebele')
print(scores_belebele)

all_languages = [
    "acm_Arab", "afr_Latn", "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "arb_Latn", "ars_Arab",
    "ary_Arab", "arz_Arab", "asm_Beng", "azj_Latn", "bam_Latn", "ben_Beng", "ben_Latn", "bod_Tibt",
    "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "ckb_Arab", "dan_Latn", "deu_Latn", "ell_Grek",
    "eng_Latn", "est_Latn", "eus_Latn", "fin_Latn", "fra_Latn", "fuv_Latn", "gaz_Latn", "grn_Latn",
    "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hin_Latn", "hrv_Latn", "hun_Latn",
    "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan",
    "kac_Latn", "kan_Knda", "kat_Geor", "kaz_Cyrl", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kin_Latn",
    "kir_Cyrl", "kor_Hang", "lao_Laoo", "lin_Latn", "lit_Latn", "lug_Latn", "luo_Latn", "lvs_Latn",
    "mal_Mlym", "mar_Deva", "mkd_Cyrl", "mlt_Latn", "mri_Latn", "mya_Mymr", "nld_Latn", "nob_Latn",
    "npi_Deva", "npi_Latn", "nso_Latn", "nya_Latn", "ory_Orya", "pan_Guru", "pbt_Arab", "pes_Arab",
    "plt_Latn", "pol_Latn", "por_Latn", "ron_Latn", "rus_Cyrl", "shn_Mymr", "sin_Latn", "sin_Sinh",
    "slk_Latn", "slv_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "srp_Cyrl",
    "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "tam_Taml", "tel_Telu", "tgk_Cyrl", "tgl_Latn",
    "tha_Thai", "tir_Ethi", "tsn_Latn", "tso_Latn", "tur_Latn", "ukr_Cyrl", "urd_Arab", "urd_Latn",
    "uzn_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "yor_Latn", "zho_Hans", "zho_Hant",
    "zsm_Latn", "zul_Latn"
]

language_mapping = pd.read_csv('csvs/language_mapping.csv')
language_mapping_index = language_mapping.set_index('FLORES-200 code')['Language'].to_dict()

names = {'ben_Latn': 'Bengali', 'hin_Latn': 'Hindi', 'npi_Latn': 'Nepali', 'sin_Latn': 'Sinhala', 'urd_Latn': 'Urdu'}

families = {'ben_Latn': 'Indo-European', 'hin_Latn': 'Indo-European', 'npi_Latn': 'Indo-European',
            'sin_Latn': 'Indo-European', 'urd_Latn': 'Indo-European'}

language_name_array = []
for lang in all_languages:
    if lang in language_mapping_index.keys():
        language_name_array.append(language_mapping_index[lang])
    else:
        language_name_array.append(names[lang])

old_results_df = pd.read_csv('csvs/old_results.csv')
lang_family = old_results_df.set_index('language_code')['language_family'].to_dict()

lang_family_array = []
for lang in all_languages:
    if lang in lang_family.keys():
        lang_family_array.append(lang_family[lang])
    else:
        lang_family_array.append(families[lang])

accuracy_array = []
for lang in all_languages:
    accuracy_array.append(round(scores_belebele[lang] / 9, 1))

r = {'ben_Latn':'Asia 2', 'hin_Latn':'Asia 2', 'npi_Latn':'Asia 2', 'sin_Latn':'Asia 2', 'urd_Latn':'Asia 2'}
region_array = []
for lang in all_languages:
    if lang in updated_region.keys():
       region_array.append(updated_region[lang])
    else:
        region_array.append(r[lang])
result_df = pd.DataFrame({
    "language_name": language_name_array,
    "language_code": all_languages,
    "language_family": lang_family_array,
    "region": region_array,
    "accuracy": accuracy_array,
})

result_df.to_csv('csvs/Belebele/belebele_results_{}.csv'.format(model_name), index=False)

df2 = pd.read_csv('csvs/Belebele/belebele_results_{}.csv'.format(model_name))
z = df2.groupby('language_family') \
    .agg({'language_name': 'size', 'accuracy': 'mean'}) \
    .rename(columns={'language_name': 'count'}) \
    .sort_values(by=['count'], ascending=False) \
    .reset_index().round(1)
z.to_csv('csvs/Belebele/belebele_results_{}_by_family.csv'.format(model_name), index=False)

z = df2.groupby('region') \
    .agg({'language_name': 'size', 'accuracy': 'mean'}) \
    .rename(columns={'language_name': 'count'}) \
    .sort_values(by=['count'], ascending=False) \
    .reset_index().round(1)
z.to_csv('csvs/Belebele/belebele_results_{}_by_region.csv'.format(model_name), index=False)

# Flores

import os
import subprocess
import json

# Dictionary to store scores
scores = {}
reference_file = reference_xx_eng_folder + "/eng_Latn.txt"

# Loop over all .txt files in the translations directory
for file in os.listdir(results_translations_folder_xx_eng):
    if file.endswith(".txt"):  # Process only .txt files
        language_name = file.replace(".txt", "")  # Extract language name
        translation_file = os.path.join(results_translations_folder_xx_eng, file)  # Full path

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
print('chrf scores for xx - Eng')
print(scores)

# Dictionary to store scores2
scores2 = {}

# Loop over all .txt files in the translations directory
for file in os.listdir(results_translations_folder_eng_xx):
    if file.endswith(".txt"):  # Process only .txt files
        language_name = file.replace(".txt", "")  # Extract language name
        translation_file = os.path.join(results_translations_folder_eng_xx, file)  # Full path
        reference_file = os.path.join(reference_eng_xx_folder, file)  # Full path

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
print('chrf scores for English - xx')
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
df.to_csv("csvs/Flores/flores_results_{}.csv".format(model_name), index=False)

###
# SIB and Flores

flores_languages = list(language_mapping_index.keys())
all_languages = list(language_mapping_index.keys())  # code
all_languages2 = list(language_mapping_index.values())  # name

import bisect

# New elements to insert
new_key = "nqo_Nkoo"
new_value = "N'ko"

# Find the correct index to insert while maintaining sorted order
index = bisect.bisect(all_languages, new_key)

# Insert in both lists at the correct position
all_languages.insert(index, new_key)
all_languages2.insert(index, new_value)

flores_df = pd.read_csv("csvs/Flores/flores_results_{}.csv".format(model_name))
old_results_df = pd.read_csv('csvs/old_results.csv')
sib_df = pd.read_csv('csvs/SIB200/sib_results_{}.csv'.format(model_name))

lang_family = old_results_df.set_index('language_code')['language_family'].to_dict()
lang_family_array = []
for lang in all_languages:
    if lang in lang_family.keys():
        lang_family_array.append(lang_family[lang])
    else:
        lang_family_array.append('')

sib_scores = sib_df.set_index('Language code')['Accuracy'].to_dict()
sib_array = []
for lang in all_languages:
    if lang in sib_scores.keys():
        sib_array.append(sib_scores[lang])
    else:
        sib_array.append('')

flores_xx_eng_scores = flores_df.set_index('FLORES-200 code')['xx - English'].to_dict()
flores_xx_eng_array = []
for lang in all_languages:
    if lang in flores_xx_eng_scores.keys():
        flores_xx_eng_array.append(flores_xx_eng_scores[lang])
    else:
        flores_xx_eng_array.append('')

flores_eng_xx_scores = flores_df.set_index('FLORES-200 code')['English - xx'].to_dict()
flores_eng_xx_array = []
for lang in all_languages:
    if lang in flores_eng_xx_scores.keys():
        flores_eng_xx_array.append(flores_eng_xx_scores[lang])
    else:
        flores_eng_xx_array.append('')

result_df = pd.DataFrame({
    "language_name": all_languages2,
    "language_code": all_languages,
    "language_family": lang_family_array,
    "region": list(updated_region.values()),
    "SIB": sib_array,
    "Flores xx-eng": flores_xx_eng_array,
    "Flores eng-xx": flores_eng_xx_array

})

result_df.to_csv('csvs/compiled/overall_results_{}.csv'.format(model_name), index=False)

df2 = pd.read_csv('csvs/compiled/overall_results_{}.csv'.format(model_name))
z = df2.groupby('language_family') \
    .agg({'language_name': 'size', 'SIB': 'mean', 'Flores xx-eng': 'mean', 'Flores eng-xx': 'mean'}) \
    .rename(columns={'language_name': 'count'}) \
    .sort_values(by=['count'], ascending=False) \
    .reset_index().round(1)
z.to_csv('csvs/compiled_by_language_family/overall_results_{}_by_family.csv'.format(model_name), index=False)

z = df2.groupby('region') \
    .agg({'language_name': 'size', 'SIB': 'mean', 'Flores xx-eng': 'mean', 'Flores eng-xx': 'mean'}) \
    .rename(columns={'language_name': 'count'}) \
    .sort_values(by=['count'], ascending=False) \
    .reset_index().round(1)
z.to_csv('csvs/compiled_by_region/overall_results_{}_by_region.csv'.format(model_name), index=False)

### LID ###

flores_l = {'afr_Latn': 'Afrikaans',
            'amh_Ethi': 'Amharic',
            'arb_Arab': 'Arabic',
            'asm_Beng': 'Assamese',
            'ast_Latn': 'Asturian',
            'azj_Latn': 'Azerbaijani',
            'bel_Cyrl': 'Belarusian',
            'ben_Beng': 'Bengali',
            'bos_Latn': 'Bosnian',
            'bul_Cyrl': 'Bulgarian',
            'cat_Latn': 'Catalan',
            'ceb_Latn': 'Cebuano',
            'ces_Latn': 'Czech',
            'ckb_Arab': 'Kurdish',
            'cym_Latn': 'Welsh',
            'dan_Latn': 'Danish',
            'deu_Latn': 'German',
            'ell_Grek': 'Greek',
            'eng_Latn': 'English',
            'est_Latn': 'Estonian',
            'fin_Latn': 'Finnish',
            'fra_Latn': 'French',
            'gaz_Latn': 'Oromo',
            'gle_Latn': 'Irish',
            'glg_Latn': 'Galician',
            'guj_Gujr': 'Gujarati',
            'hau_Latn': 'Hausa',
            'heb_Hebr': 'Hebrew',
            'hin_Deva': 'Hindi',
            'hrv_Latn': 'Croatian',
            'hun_Latn': 'Hungarian',
            'hye_Armn': 'Armenian',
            'ibo_Latn': 'Igbo',
            'ind_Latn': 'Indonesian',
            'isl_Latn': 'Icelandic',
            'ita_Latn': 'Italian',
            'jav_Latn': 'Javanese',
            'jpn_Jpan': 'Japanese',
            'kam_Latn': 'Kamba',
            'kan_Knda': 'Kannada',
            'kat_Geor': 'Georgian',
            'kaz_Cyrl': 'Kazakh',
            'kea_Latn': 'Kabuverdianu',
            'khk_Cyrl': 'Mongolian',
            'khm_Khmr': 'Khmer',
            'kor_Hang': 'Korean',
            'lao_Laoo': 'Lao',
            'lin_Latn': 'Lingala',
            'lit_Latn': 'Lithuanian',
            'ltz_Latn': 'Luxembourgish',
            'lug_Latn': 'Ganda',
            'luo_Latn': 'Luo',
            'lvs_Latn': 'Latvian',
            'mal_Mlym': 'Malayalam',
            'mar_Deva': 'Marathi',
            'mkd_Cyrl': 'Macedonian',
            'mlt_Latn': 'Maltese',
            'mri_Latn': 'Maori',
            'mya_Mymr': 'Burmese',
            'nld_Latn': 'Dutch',
            'nob_Latn': 'Norwegian',
            'npi_Deva': 'Nepali',
            'nso_Latn': 'Sotho',
            'nya_Latn': 'Nyanja',
            'oci_Latn': 'Occitan',
            'ory_Orya': 'Odia',
            'pan_Guru': 'Punjabi',
            'pbt_Arab': 'Pashto',
            'pes_Arab': 'Persian',
            'pol_Latn': 'Polish',
            'por_Latn': 'Portuguese',
            'ron_Latn': 'Romanian',
            'rus_Cyrl': 'Russian',
            'slk_Latn': 'Slovak',
            'slv_Latn': 'Slovenian',
            'smo_Latn': 'Samoan',
            'sna_Latn': 'Shona',
            'snd_Arab': 'Sindhi',
            'som_Latn': 'Somali',
            'spa_Latn': 'Spanish',
            'srp_Cyrl': 'Serbian',
            'swe_Latn': 'Swedish',
            'swh_Latn': 'Swahili',
            'tam_Taml': 'Tamil',
            'tel_Telu': 'Telugu',
            'tgk_Cyrl': 'Tajik',
            'tgl_Latn': 'Tagalog',
            'tha_Thai': 'Thai',
            'tur_Latn': 'Turkish',
            'ukr_Cyrl': 'Ukrainian',
            'umb_Latn': 'Umbundu',
            'urd_Arab': 'Urdu',
            'uzn_Latn': 'Uzbek',
            'vie_Latn': 'Vietnamese',
            'wol_Latn': 'Wolof',
            'xho_Latn': 'Xhosa',
            'yor_Latn': 'Yoruba',
            'zho_Hans': 'Chinese',
            'zho_Hant': 'Chinese',
            'zsm_Latn': 'Malay',
            'zul_Latn': 'Zulu'}

redo_l = {
    'ace_Arab': ['Malay', 'Jawi'],
    'aka_Latn': ['Akan', 'Twi'],
    'arb_Arab': ['Arabic', 'العربية'],
    'acm_Arab': ['Arabic', 'العربية'],
    'acq_Arab': ['Arabic', 'العربية'],
    'aeb_Arab': ['Arabic', 'العربية'],
    'ajp_Arab': ['Arabic', 'العربية'],
    'apc_Arab': ['Arabic', 'العربية'],
    'arb_Latn': ['Arabic', 'العربية'],
    'ars_Arab': ['Arabic', 'العربية'],
    'ary_Arab': ['Arabic', 'العربية'],
    'arz_Arab': ['Arabic', 'العربية'],
    'asm_Beng': ['Assamese', 'অসমীয়া'],
    'azb_Arab': ['Azerbaijani', 'Azeri'],
    'bak_Cyrl': ['Bashkir', 'Башкорт теле'],
    'bam_Latn': ['Bambara', 'Bamanankan'],
    'bug_Latn': ['Buginese', 'Bugis'],
    'bos_Latn': ['Bosnian', 'Croatian', 'Serbian'],
    'fuv_Latn': ['Fulfulde', 'Fula', 'Pulaar'],
    'kac_Latn': ['Jingpho', 'Jinghpaw', 'Kachin'],
    'kbp_Latn': ['Kabiyè', 'Kabiye', 'Kabɩyɛ'],
    'kea_Latn': ['Kabuverdianu', 'Creole', 'Kriol'],
    'kik_Latn': ['Kikuyu', 'Gikuyu', 'Gĩkũyũ'],
    'kon_Latn': ['Kikongo', 'Kongo'],
    'ltg_Latn': ['Latgalian', 'Latvian'],
    'lua_Latn': ['Luba-Kasai', 'Luba'],
    'mai_Deva': ['Maithili', 'मैथिली'],
    'mri_Latn': ['Maori', 'Māori'],
    'mni_Beng': ['Meitei', 'Manipuri'],
    'mos_Latn': ['Mossi', 'Mooré', 'Moore'],
    'nso_Latn': ['Sotho', 'Sepedi', 'Setswana'],
    'nya_Latn': ['Nyanja', 'Chewa'],
    'prs_Arab': ['Dari', 'Persian'],
    'slv_Latn': ['Slovenian', 'Slovene'],
    'tgl_Latn': ['Tagalog', 'Filipino'],
    'twi_Latn': ['Akan', 'Twi'],
    'zho_Hant': ['Chinese']
}
redo_languages = list(redo_l.keys())

remaining_l = {'ace_Arab': 'Acehnese',
               'ace_Latn': 'Acehnese',
               'acm_Arab': 'Arabic',
               'acq_Arab': 'Arabic',
               'aeb_Arab': 'Arabic',
               'ajp_Arab': 'Arabic',
               'aka_Latn': 'Akan',
               'als_Latn': 'Albanian',
               'apc_Arab': 'Arabic',
               'arb_Latn': 'Arabic',
               'ars_Arab': 'Arabic',
               'ary_Arab': 'Arabic',
               'arz_Arab': 'Arabic',
               'awa_Deva': 'Awadhi',
               'ayr_Latn': 'Aymara',
               'azb_Arab': 'Azerbaijani',
               'bak_Cyrl': 'Bashkir',
               'bam_Latn': 'Bambara',
               'ban_Latn': 'Balinese',
               'bem_Latn': 'Bemba',
               'bho_Deva': 'Bhojpuri',
               'bjn_Arab': 'Banjar',
               'bjn_Latn': 'Banjar',
               'bod_Tibt': 'Tibetan',
               'bug_Latn': 'Buginese',
               'cjk_Latn': 'Chokwe',
               'crh_Latn': 'Tatar',
               'dik_Latn': 'Dinka',
               'dyu_Latn': 'Dyula',
               'dzo_Tibt': 'Dzongkha',
               'epo_Latn': 'Esperanto',
               'eus_Latn': 'Basque',
               'ewe_Latn': 'Ewe',
               'fao_Latn': 'Faroese',
               'fij_Latn': 'Fijian',
               'fon_Latn': 'Fon',
               'fur_Latn': 'Friulian',
               'fuv_Latn': 'Fulfulde',
               'gla_Latn': 'Gaelic',
               'grn_Latn': 'Guarani',
               'hat_Latn': 'Creole',
               'hne_Deva': 'Chhattisgarhi',
               'ilo_Latn': 'Ilocano',
               'kab_Latn': 'Kabyle',
               'kac_Latn': 'Jingpho',
               'kas_Arab': 'Kashmiri',
               'kas_Deva': 'Kashmiri',
               'kbp_Latn': 'Kabiyè',
               'kik_Latn': 'Kikuyu',
               'kin_Latn': 'Kinyarwanda',
               'kir_Cyrl': 'Kyrgyz',
               'kmb_Latn': 'Kimbundu',
               'kmr_Latn': 'Kurdish',
               'knc_Arab': 'Kanuri',
               'knc_Latn': 'Kanuri',
               'kon_Latn': 'Kikongo',
               'lij_Latn': 'Ligurian',
               'lim_Latn': 'Limburgish',
               'lmo_Latn': 'Lombard',
               'ltg_Latn': 'Latgalian',
               'lua_Latn': 'Luba-Kasai',
               'lus_Latn': 'Mizo',
               'mag_Deva': 'Magahi',
               'mai_Deva': 'Maithili',
               'min_Arab': 'Minangkabau',
               'min_Latn': 'Minangkabau',
               'mni_Beng': 'Meitei',
               'mos_Latn': 'Mossi',
               'nno_Latn': 'Norwegian',
               'nus_Latn': 'Nuer',
               'pag_Latn': 'Pangasinan',
               'pap_Latn': 'Papiamento',
               'plt_Latn': 'Malagasy',
               'prs_Arab': 'Dari',
               'quy_Latn': 'Quechua',
               'run_Latn': 'Rundi',
               'sag_Latn': 'Sango',
               'san_Deva': 'Sanskrit',
               'sat_Olck': 'Santali',
               'scn_Latn': 'Sicilian',
               'shn_Mymr': 'Shan',
               'sin_Sinh': 'Sinhala',
               'sot_Latn': 'Sotho',
               'srd_Latn': 'Sardinian',
               'ssw_Latn': 'Swati',
               'sun_Latn': 'Sundanese',
               'szl_Latn': 'Silesian',
               'taq_Latn': 'Tamasheq',
               'taq_Tfng': 'Tamasheq',
               'tat_Cyrl': 'Tatar',
               'tir_Ethi': 'Tigrinya',
               'tpi_Latn': 'Pisin',
               'tsn_Latn': 'Tswana',
               'tso_Latn': 'Tsonga',
               'tuk_Latn': 'Turkmen',
               'tum_Latn': 'Tumbuka',
               'twi_Latn': 'Twi',
               'tzm_Tfng': 'Tamazight',
               'uig_Arab': 'Uyghur',
               'vec_Latn': 'Venetian',
               'war_Latn': 'Waray',
               'ydd_Hebr': 'Yiddish',
               'yue_Hant': 'Chinese'}

# redo those with valid reasons

for language_code in redo_languages:
    correct_answers = redo_l[language_code]
    result_list = []

    reply_path = os.path.join(reply_dir_lid, "{}.csv".format(language_code))
    df = pd.read_csv(reply_path)

    for i in range(1012):
        reply = df['reply'][i]
        result = "invalid"
        for correct_answer in correct_answers:  # any answer matches it's a match.
            if correct_answer.lower() in reply.lower():
                if language_code in list(flores_l.keys()):
                    result = flores_l[language_code]
                else:
                    result = remaining_l[language_code]
                break
        result_list.append(result)

    df = pd.DataFrame({
        "answer": result_list
    })

    # Save to CSV
    df.to_csv("{}/{}.csv".format(results_dir_lid, language_code), index=False)

all_languages = list(language_mapping_index.keys())

scores_lid = {}

for file in os.listdir(results_dir_lid):
    if file.endswith(".csv"):  # Process only .txt files
        language_code = file.replace(".csv", "")
        if language_code not in all_languages:
            continue
        if language_code in list(flores_l.keys()):
            correct_answer = flores_l[language_code]
        else:
            correct_answer = remaining_l[language_code]
        file_path = os.path.join(results_dir_lid, file)
        df = pd.read_csv(file_path)

        matches = (df['answer'] == correct_answer).sum()
        scores_lid[language_code] = matches
print('scores_lid')
print(scores_lid)

keys = [k for k, v in scores_lid.items() if v < 10]
print('low_scores_lid')
print(keys)

language_name_array = []
for lang in all_languages:
    language_name_array.append(language_mapping_index[lang])

old_results_df = pd.read_csv('csvs/old_results.csv')
lang_family = old_results_df.set_index('language_code')['language_family'].to_dict()

lang_family_array = []
for lang in all_languages:
    lang_family_array.append(lang_family[lang])

results_array = []
for lang in all_languages:
    results_array.append(round(scores_lid[lang] / 10.12, 1))

region_array = []
for lang in all_languages:
    region_array.append(updated_region[lang])

result_df = pd.DataFrame({
    "language_name": language_name_array,
    "language_code": all_languages,
    "language_family": lang_family_array,
    "region": region_array,
    "accuracy": results_array
})

result_df.to_csv('csvs/LID/lid_results_{}.csv'.format(model_name), index=False)

z = result_df.groupby('region') \
    .agg({'language_name': 'size', 'accuracy': 'mean'}) \
    .rename(columns={'language_name': 'count'}) \
    .sort_values(by=['count'], ascending=False) \
    .reset_index().round(1)
z.to_csv('csvs/LID/lid_results_{}_by_region.csv'.format(model_name), index=False)

z = result_df.groupby('language_family') \
    .agg({'language_name': 'size', 'accuracy': 'mean'}) \
    .rename(columns={'language_name': 'count'}) \
    .sort_values(by=['count'], ascending=False) \
    .reset_index().round(1)
z.to_csv('csvs/LID/lid_results_{}_by_family.csv'.format(model_name), index=False)

### leaderboard csv
import pandas as pd

lid_df = pd.read_csv('csvs/LID/lid_results_{}_by_region.csv'.format(model_name))
sib_df = pd.read_csv('csvs/compiled_by_region/overall_results_{}_by_region.csv'.format(model_name))
belebele_df = pd.read_csv('csvs/Belebele/belebele_results_{}_by_region.csv'.format(model_name))
nli_df = pd.read_csv('csvs/GlobalNLI/nli_results_{}_by_region.csv'.format(model_name))

lid_by_region = lid_df.set_index('region')['accuracy'].to_dict()
sib_by_region = sib_df.set_index('region')['SIB'].to_dict()
belebele_by_region = belebele_df.set_index('region')['accuracy'].to_dict()
nli_by_region = nli_df.set_index('region')['accuracy'].to_dict()
flores_xx_eng_by_region = sib_df.set_index('region')['Flores xx-eng'].to_dict()
flores_eng_xx_by_region = sib_df.set_index('region')['Flores eng-xx'].to_dict()

order = [
    "Africa",
    "Americas/Oceania",
    "Asia (S)",
    "Asia (SE)",
    "Asia (W, C)",
    "Asia (E)",
    "Europe (W, N, S)",
    "Europe (E)"
]

region_map = {
    "Africa": "Africa",
    "Americas/Oceania": "Americas/Oceania",
    "Asia (S)": "Asia 2",
    "Asia (SE)": "Asia 3",
    "Asia (W, C)": "Asia 1",
    "Asia (E)": "Asia 4",
    "Europe (W, N, S)": "Europe 1",
    "Europe (E)": "Europe 2"
}

result_df = pd.DataFrame({
    "Region": order,
    "LID": [lid_by_region[region_map[reg]] for reg in order],
    "TC": [sib_by_region[region_map[reg]] for reg in order],
    "RC-QA": [belebele_by_region[region_map[reg]] for reg in order],
    "NLI": [nli_by_region[region_map[reg]] for reg in order],
    "MT (xx-en)": [flores_xx_eng_by_region[region_map[reg]] for reg in order],
    "MT (en-xx)": [flores_eng_xx_by_region[region_map[reg]] for reg in order],
})

# for microaverages:
lid_df = pd.read_csv('csvs/LID/lid_results_{}.csv'.format(model_name))
sib_df = pd.read_csv('csvs/compiled/overall_results_{}.csv'.format(model_name))
belebele_df = pd.read_csv('csvs/Belebele/belebele_results_{}.csv'.format(model_name))
nli_df = pd.read_csv('csvs/GlobalNLI/nli_results_{}.csv'.format(model_name))

micro_averages = ['Average (Micro)', lid_df['accuracy'].mean().round(1), sib_df['SIB'].mean().round(1),
                  belebele_df['accuracy'].mean().round(1), nli_df['accuracy'].mean().round(1),
                  sib_df['Flores xx-eng'].mean().round(1), sib_df['Flores eng-xx'].mean().round(1)]

result_df.loc[len(result_df)] = micro_averages

result_df.to_csv('csvs/compiled/leaderboard_{}.csv'.format(model_name),index=False)
