# import random
# import os
import yaml

# from typing import List
import pandas as pd

from multiple_choice import get_model_answer_multiple_options
from multiple_choice import compare_answers

from rag import get_answer_from_local_ollama_context
from rag import get_evaluation_score_context

from qa import get_answer_from_local_ollama
from qa import get_evaluation_score, calculate_rouge_score, calculate_bleu_score, calculate_levenshtein_score


"""
    This is the main file for executing the code in a single run to obtain the result for the YAML version.
"""



with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


metadata = config['metadata']
dataset_files = config['dataset_files']
results_file = config['output']['results_file']



def get_benchmark_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        ending = ending + '.xlsx'

        if filename.endswith(ending):
            return benchmark_type
    raise ValueError(f"Filename {filename} does not match any known benchmark type")

def handle_qa(question, actual_answer, model):
    predicted_answer = get_answer_from_local_ollama(model, question)
    # score = min(max(int(float(get_evaluation_score(question, predicted_answer, actual_answer))), 0), 100)
    score = (0.25 * int(float(get_evaluation_score(question, actual_answer, predicted_answer)))) + calculate_bleu_score(actual_answer, predicted_answer) + calculate_rouge_score(actual_answer, predicted_answer) + calculate_levenshtein_score(actual_answer, predicted_answer)
    return score

def handle_multiple_choice(question, options, correct_option, model):
    predicted_option = get_model_answer_multiple_options(question, options=options, model=model, dstype='mc')
    # print(predicted_option)
    score = compare_answers(actual_answer=correct_option, predicted_answer=predicted_option)
    return score

def handle_context_qa(question, context, actual_answer, model):
    predicted_answer = get_answer_from_local_ollama_context(model, question, context)
    score = (0.25 * int(float(get_evaluation_score_context(question, actual_answer, predicted_answer)))) + calculate_bleu_score(actual_answer, predicted_answer) + calculate_rouge_score(actual_answer, predicted_answer) + calculate_levenshtein_score(actual_answer, predicted_answer)
    return score

def handle_topic_classification(question, topic_options, correct_topic, model):
    predicted_topic = get_model_answer_multiple_options(question=question, model=model, options=topic_options, dstype='tc')
    print(predicted_topic)
    score = compare_answers(actual_answer=correct_topic, predicted_answer=predicted_topic)
    return score

def handle_arc(question, options, correct_answer, model):
    predicted_option = get_model_answer_multiple_options(question, options=options, model=model, dstype='arc')
    # print(predicted_option)
    score = compare_answers(actual_answer=correct_answer, predicted_answer=predicted_option)
    return score

def run_benchmark(model_name, benchmark_type, df, results):
    scores = []
    if benchmark_type == "QA":
        for index, row in df.iterrows():
            question = row['Sual']
            actual_answer = row['Cavab']
            score = handle_qa(question, actual_answer, model_name)
            scores.append(score)
    
    elif benchmark_type == "Reshad":
        for index, row in df.iterrows():
            question = row['text']
            options = row['options']
            correct_option = row['answer']
            score = handle_topic_classification(question, options, correct_option, model_name)
            scores.append(score)

    elif benchmark_type == "ContextQA":
        for index, row in df.iterrows():
            question = row['question']
            context = row['context']
            actual_answer = row['answer']
            score = handle_context_qa(question, context, actual_answer, model_name)
            scores.append(score)

    elif benchmark_type == "Arzuman":
        for index, row in df.iterrows():
            question = row['text']
            topic_options = row['options']
            correct_topic = row['answer']
            score = handle_multiple_choice(question, topic_options, correct_topic, model_name)
            scores.append(score)

    elif benchmark_type == "ARC":
        for index, row in df.iterrows():
            question = row['Azerbaijani_q']
            options_txt = row['choices']
            array = pd.array
            options_dict = eval(options_txt)
            options = options_dict['az_choices'].tolist()
            # print(options)
            correct_answer = row['answerKey']
            score = handle_arc(question, options, correct_answer, model_name)
            scores.append(score)

    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    if scores:
        average_score = sum(scores) / len(scores)
        results.loc[model_name, benchmark_type] = average_score

results = pd.DataFrame(columns=metadata['benchmark_types'].keys(), index=metadata['supported_models'])

for file in dataset_files:
    benchmark_type = get_benchmark_from_filename(file, metadata)
    print(f"Running {benchmark_type} benchmark for file: {file}")
    
    df = pd.read_excel(file)
    df = df[:2]  
    print(df)

    for model_name in metadata['supported_models']:
        print(f"Running {benchmark_type} for model {model_name}")
        run_benchmark(model_name, benchmark_type, df, results)

print("\nAverage Scores:\n", results)

# results.to_excel("benchmark_results.xlsx", engine='openpyxl')
results.to_excel(results_file)


