Some pointers here for adding a new model.

Follow the code for gemma3n to see the most recent way of running everything. In code/ you can add a
 file for each task.

Run things from the root of the repository.

Run the experiments in any order for each of the tasks, then look at the code in compiling_results
for gemma3n (results_gemma3n.py ) to see how to get different csvs. This compiling is reusable (as is),
as long as the experiments have been run in the same convention. Just pass the model name as a parameter
to the script. Some file names are by convention. This compilation file generates multiple csvs for different
tasks, including those for language families, regions, and the final csv that goes in the leaderboard.
SIB and Flores are compiled together and their region based results go in csvs/compiled




