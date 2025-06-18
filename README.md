# mSTEB

This repository contains code and results for mSTEB:

https://arxiv.org/pdf/2506.08400

The code is organized in directory code/

Results of evaluation are in results/

CSV files containing summary of results for individual tasks and other aggregate tasks are in csvs/


If running Gemini/OpenAI code export API key in the right environment

export GEMINI_KEY = 'your_api_key'
export OPENAI_API_KEY = 'your_api_key'

provide other arguments while running the script: 

python code/Belebele/belebele_gemini2.py --results_folder='../results/Belebele/belebele_results_gemini2' --results_reply_folder='../results/Belebele/belebele_replies_gemini2

results_folder has per language model results
results_reply_folder has per language model replies (useful for debugging)
results_csv_folder has the compiled results for the task

