# mSTEB

This repository contains code and results for **mSTEB**: multilingual Speech and Text Evaluation Benchmark

 **Paper**: https://arxiv.org/pdf/2506.08400

## Repository Structure: 

The code is organized in directory code/

Results of evaluation are in results/

CSV files containing summary of results for individual tasks and other aggregate tasks are in csvs/

## Running code:

If running Gemini/OpenAI code export API key in the right environment

```
export GEMINI_KEY='your_api_key'
export OPENAI_API_KEY='your_api_key'
```

provide other arguments while running the script: 

```
python code/Belebele/belebele_gemini2.py \
  --results_folder='../results/Belebele/belebele_results_gemini2' \
  --results_reply_folder='../results/Belebele/belebele_replies_gemini2'
```

_results_folder_ has per language model results 

_results_reply_folder_ has per language model replies (useful for debugging)

_results_csv_folder_ has the compiled results for the task

