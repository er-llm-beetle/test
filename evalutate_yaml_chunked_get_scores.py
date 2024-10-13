# import random
# import os
import yaml

# from typing import List
import pandas as pd
import traceback 

# from multiple_choice import compare_answers, compare_answers_and_save
from multiple_choice_w_api import compare_answers, compare_answers_and_save

# from rag import get_evaluation_score_context
from rag_w_api import get_evaluation_score_context

# from qa import get_evaluation_score, calculate_rouge_score, calculate_bleu_score, calculate_levenshtein_score
from qa_w_api import get_evaluation_score, calculate_rouge_score, calculate_bleu_score, calculate_levenshtein_score


# version 1
# with open('config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# version 2
with open('config_w_api.yaml', 'r') as file:
    config = yaml.safe_load(file)


metadata = config['metadata']
dataset_files = config['dataset_files']
results_file = config['output']['results_file']


# Score handlers
def handle_qa_score(question, actual_answer, predicted_answer, benchmark_type, model_name):
    if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
        return 0
    
    score = (0.25 * int(float(get_evaluation_score(question, actual_answer, predicted_answer, benchmark_type=benchmark_type, model_name=model_name)))) \
            + calculate_bleu_score(actual_answer, predicted_answer) \
            + calculate_rouge_score(actual_answer, predicted_answer) \
            + calculate_levenshtein_score(actual_answer, predicted_answer)
    return score

def handle_context_qa_score(question, context, actual_answer, predicted_answer, benchmark_type, model_name):
    if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
        return 0

    score = (0.25 * int(float(get_evaluation_score_context(question, actual_answer, predicted_answer, benchmark_type=benchmark_type, model_name=model_name)))) \
            + calculate_bleu_score(actual_answer, predicted_answer) \
            + calculate_rouge_score(actual_answer, predicted_answer) \
            + calculate_levenshtein_score(actual_answer, predicted_answer)
    return score

def handle_multiple_choice_score(actual_answer, predicted_answer, benchmark_type, model_name):
    # return compare_answers(actual_answer=actual_answer, predicted_answer=predicted_answer)
    return compare_answers_and_save(actual_answer=actual_answer, predicted_answer=predicted_answer, benchmark_type=benchmark_type, model_name=model_name)

def handle_topic_classification_score(correct_topic, predicted_topic, benchmark_type, model_name):
    # return compare_answers(actual_answer=correct_topic, predicted_answer=predicted_topic)
    return compare_answers_and_save(actual_answer=correct_topic, predicted_answer=predicted_topic, benchmark_type=benchmark_type, model_name=model_name)

def handle_arc_score(correct_answer, predicted_option, benchmark_type, model_name):
    # return compare_answers(actual_answer=correct_answer, predicted_answer=predicted_option)
    return compare_answers_and_save(actual_answer=correct_answer, predicted_answer=predicted_option, benchmark_type=benchmark_type, model_name=model_name)


# Calculate Scores Function with Error Handling
def calculate_scores(predictions_file, benchmark_type, model_name):
    scores = []
    try:
        predictions_df = pd.read_excel(predictions_file)

        if benchmark_type == "QA":
            for index, row in predictions_df.iterrows():
                score = handle_qa_score(row['Question'], row['Correct Answer'], row['Predicted Answer'], benchmark_type, model_name)
                scores.append(score)

        elif benchmark_type == "ContextQA":
            for index, row in predictions_df.iterrows():
                score = handle_context_qa_score(row['Question'], row['Context'], row['Correct Answer'], row['Predicted Answer'], benchmark_type, model_name)
                scores.append(score)

        elif benchmark_type == "Arzuman":
            for index, row in predictions_df.iterrows():
                score = handle_multiple_choice_score(row['Correct Answer'], row['Predicted Option'], benchmark_type, model_name)
                scores.append(score)

        elif benchmark_type == "Reshad":
            for index, row in predictions_df.iterrows():
                score = handle_topic_classification_score(row['Correct Answer'], row['Predicted Topic'], benchmark_type, model_name)
                scores.append(score)

        elif benchmark_type == "ARC":
            for index, row in predictions_df.iterrows():
                score = handle_arc_score(row['Correct Answer'], row['Predicted Option'], benchmark_type, model_name)
                scores.append(score)

        elif benchmark_type == "QMC":
            for index, row in predictions_df.iterrows():
                score = handle_multiple_choice_score(row['Correct Answer'], row['Predicted Option'], benchmark_type, model_name)
                scores.append(score)

        elif benchmark_type == "mMC":
            for index, row in predictions_df.iterrows():
                score = handle_multiple_choice_score(row['Correct Answer'], row['Predicted Option'], benchmark_type, model_name)
                scores.append(score)

        elif "kMC" in benchmark_type:
            for index, row in predictions_df.iterrows():
                score = handle_multiple_choice_score(row['Correct Answer'], row['Predicted Option'], benchmark_type, model_name)
                scores.append(score)

        # Calculate and return average score
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0

    except Exception as e:
        print(f"Error occurred while calculating scores: {str(e)}")
        traceback.print_exc() 


# Main function to get scores based on stored answers with error handling
def run_benchmark_get_scores(model_name, benchmark_type):
    try:
        print('run benchmark get scores started')
        predictions_file = f"{benchmark_type}_{model_name}_predictions.xlsx"
        print("predictions_file", predictions_file)
        # average_score = calculate_scores(predictions_file, benchmark_type)
        average_score = calculate_scores(predictions_file, benchmark_type, model_name) # v2

        if average_score is not None:
            print(f"Average Score for {model_name} on {benchmark_type}: {average_score}")
        return average_score
    except Exception as e:
        print(f"Error while calculating scores for {benchmark_type} with model {model_name}: {str(e)}")
        traceback.print_exc()

