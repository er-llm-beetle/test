# import random
# import os
import yaml

# from typing import List
import pandas as pd
import traceback 

# from multiple_choice import get_model_answer_multiple_options
# from multiple_choice import compare_answers

# from rag import get_answer_from_local_ollama_context
# from rag import get_evaluation_score_context

# from qa_quality import get_answer_from_local_ollama
# from qa_quality import get_evaluation_score, calculate_rouge_score, calculate_bleu_score, calculate_levenshtein_score

from evalutate_yaml_chunked_get_answers import run_benchmark_store_answers
from evalutate_yaml_chunked_get_scores import run_benchmark_get_scores



"""
    This code serves as the main file for chunked execution, obtaining answers separately and evaluating scores separately, as well as for storing answers, scores, and etc.
"""


# version 1
# with open('config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# version 2
with open('config_w_api.yaml', 'r') as file:
    config = yaml.safe_load(file)


metadata = config['metadata']
dataset_files = config['dataset_files']
results_file = config['output']['results_file']



def get_benchmark_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        print("GET benchmark", benchmark_type, ending, filename)
        ending = ending + '.xlsx'
        if filename.endswith(ending):
            return benchmark_type, ending[1:]
    raise ValueError(f"Filename {filename} does not match any known benchmark type")



# Call this function to run Step 1: Get answers (predictions) and store them in file
def run_step_1_store_answers():
    print("Running Step 1: Store Answers")

    for file in dataset_files:
        print(file)
        benchmark_type, ending = get_benchmark_from_filename(file, metadata)
        df = pd.read_excel(file)

        # Limit for testing (Optional)
        # df = df[:2]

        # v1 (without api)
        # for model_name in metadata['supported_models']:
        #     print(f"Storing answers for {benchmark_type} benchmark with model {model_name}")
        #     try:
        #         run_benchmark_store_answers(model_name, benchmark_type, df)
        #     except Exception as e:
        #         print(f"Error during answer storage: {str(e)}")
        #         traceback.print_exc()

        # v2 (with api and changed yaml structure)
        for mode, models in metadata['supported_models'].items():
            for model_name in models:
                print("\nmode:", mode)
                print("model_name:", model_name)
                try:
                    # if mode == "api":
                        # Handle the API model-specific logic
                        print(f"Running {benchmark_type} benchmark with {model_name} via {mode} model")
                        run_benchmark_store_answers(model_name, benchmark_type, df, mode, ending, file)
                    # elif mode == "local":
                    #     # Handle the local model-specific logic
                    #     print(f"Running benchmark with {model_name} locally")
                    #     run_benchmark_store_answers(model_name, benchmark_type, df, api=False)
                    # else:
                        # print(f"Unknown mode: {mode}")
                except Exception as e:
                    print(f"Error during answer storage for model {model_name}: {str(e)}")
                    traceback.print_exc()


# Call this function to run Step 2: Calculate scores from the stored files (be sure that files exists)
def run_step_2_calculate_scores():
    print("Running Step 2: Calculate Scores")


    # version 1 (local only)
    # # DataFrame to store final results
    # results = pd.DataFrame(columns=metadata['benchmark_types'].keys(), index=metadata['supported_models'])

    # for model_name in metadata['supported_models']:
    #     for benchmark_type in metadata['benchmark_types'].keys():
    #         try:
    #             print(f"Running Calculate Scores for model: {model_name}, benchmark: {benchmark_type} ")
    #             average_score = run_benchmark_get_scores(model_name, benchmark_type)
    #             results.loc[model_name, benchmark_type] = average_score
    #         except Exception as e:
    #             print(f"Error during score calculation: {str(e)}")
    #             traceback.print_exc()


    # version 2 (with local and api incl.)
    results = pd.DataFrame(columns=metadata['benchmark_types'].keys())

    # Loop through the models under both 'local' and 'api'
    for mode, models in metadata['supported_models'].items():
        for model_name in models:
            for benchmark_type in metadata['benchmark_types'].keys():
                try:
                    print(f"Running Calculate Scores for model: {model_name}, benchmark: {benchmark_type}, mode: {mode}")
                    # run_benchmark_get_scores is a function that calculates and returns average score
                    average_score = run_benchmark_get_scores(model_name, benchmark_type)
                    # Store the result in the dataframe under the correct model and benchmark type
                    results.loc[model_name, benchmark_type] = average_score
                    print(results, '\n')
                except Exception as e:
                    print(f"Error during score calculation: {str(e)}")
                    traceback.print_exc()


    # Save the results after calculating all scores
    print("\nAverage Scores:\n", results)
    results.to_excel(results_file)
    print(f"Results saved to {results_file}")



# Combine both steps in one function call
def run_both_steps():
   # First, run Step 1: Get answers (predictions)
    # print("FIRST STEP (Get answers)")
    run_step_1_store_answers()

    # Second, run Step 2: Calculate scores
    # print("SECOND STEP (Calculate scores)")
    run_step_2_calculate_scores()


# Main function to trigger both steps
if __name__ == '__main__':
    run_both_steps()

