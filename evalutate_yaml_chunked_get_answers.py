import os
import yaml

import pandas as pd
import traceback  


# from multiple_choice import get_model_answer_multiple_options
from multiple_choice_w_api import get_model_answer_multiple_options

# from rag import get_answer_from_local_ollama_context
from rag_w_api import get_answer_from_local_ollama_context

# from qa import get_answer_from_local_ollama
from qa_w_api import get_answer_from_local_ollama


# version 1
# with open('config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# version 2
with open('config_w_api.yaml', 'r') as file:
    config = yaml.safe_load(file)

metadata = config['metadata']
dataset_files = config['dataset_files']
results_file = config['output']['results_file']



def get_subtext(text, start_txt, end_txt): # check it
    # Extracting the relevant part
    start = text.find(start_txt) + len(start_txt)
    end = text.find(end_txt)
    extracted_text = text[start:end]

    return extracted_text


def get_benchmark_ending_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        ending = ending
        if filename.endswith(ending):
            return ending[:1]


def get_benchmark_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        ending = ending + '.xlsx'
        if filename.endswith(ending):
            return benchmark_type
    raise ValueError(f"Filename {filename} does not match any known benchmark type")

# Prediction handlers
def handle_qa_prediction(question, model, mode):
    return get_answer_from_local_ollama(model, question, mode=mode)

def handle_context_qa_prediction(question, context, model, mode):
    return get_answer_from_local_ollama_context(model, question, context, mode=mode)

def handle_multiple_choice_prediction(question, options, model, mode):
    return get_model_answer_multiple_options(question, options=options, model=model, dstype='mc', mode=mode)

def handle_topic_classification_prediction(question, options, model, mode):
    return get_model_answer_multiple_options(question=question, model=model, options=options, dstype='tc', mode=mode)

def handle_arc_prediction(question, options, model, mode):
    return get_model_answer_multiple_options(question, options=options, model=model, dstype='arc', mode=mode)

def handle_quad_prediction(question, context, options, model, mode):
    return get_model_answer_multiple_options(question, options, model, dstype='qmc', mode=mode, context=context)

def handle_gsm8k_prediction(question, options, model, mode):
    return get_model_answer_multiple_options(question, options=options, model=model, dstype='mmc', mode=mode)


def handle_mmlu_aze_prediction(question, options, model, mode, ending, file):
    if ending in file:

        subtext = get_subtext(file, "testler_", "_kmc")

        return get_model_answer_multiple_options(question, options=options, model=model, dstype=f'kmc_{subtext}', mode=mode)
    


# Store Predictions Function with Error Handling
def store_predictions(df, benchmark_type, model_name, mode, ending, file):
    predictions = []
    try:
        start_index = 0
        output_filename = f"{benchmark_type}_{model_name}_predictions.xlsx"

        if "kmc" in file:
            subtext = get_subtext(file, "testler_", "_kmc")
    
            output_filename = f"{benchmark_type}_{subtext}_{model_name}_predictions.xlsx"



        # Check if the file exists and read existing data
        if os.path.exists(output_filename):
            existing_df = pd.read_excel(output_filename)
            
            # Define columns based on benchmark type for existing data
            if benchmark_type == "QA":
                columns_to_keep = ['Question', 'Correct Answer', 'Predicted Answer']
            elif benchmark_type == "ContextQA":
                columns_to_keep = ['Question', 'Context', 'Correct Answer', 'Predicted Answer']
            elif benchmark_type == "Arzuman":
                columns_to_keep = ['Question', 'Correct Answer', 'Predicted Option']
            elif benchmark_type == "Reshad":
                columns_to_keep = ['Question', 'Correct Answer', 'Predicted Topic']
            elif benchmark_type == "ARC":
                columns_to_keep = ['Question', 'Correct Answer', 'Predicted Option']
            elif benchmark_type == "kMC":
                columns_to_keep = ['Question', 'Correct Answer', 'Predicted Option']
            elif benchmark_type == "mMC":
                columns_to_keep = ['Question', 'Correct Answer', 'Predicted Option']
            elif benchmark_type == "QMC":
                columns_to_keep = ['Question', 'Context', 'Correct Answer', 'Predicted Option']
            else:
                raise ValueError("Unknown benchmark type.")

            # Filter the existing DataFrame to keep only relevant columns
            existing_df = existing_df[columns_to_keep]            
            
            if existing_df[existing_df.columns[-1]][~existing_df[existing_df.columns[-1]].isin(['Error', 'Long answer'])].nunique(dropna=True) > 0:
                start_index = len(existing_df)  # Start from the next index (default case) # 

            # Iterate through the DataFrame to find the start index (when happened error at the point) #
            for i in range(1, len(existing_df)):
                if existing_df.loc[i, existing_df.columns[-1]].lower() == 'error':
                    if existing_df.loc[i - 1, existing_df.columns[-1]].lower() != 'error':
                        start_index = i
                        break

            # Manually define starting index # v3
            # start_index = 2

            print("Start index:", start_index)

            predictions = existing_df.iloc[:start_index].values.tolist()  # Keep existing predictions
        else:
            start_index = 0  # Start from the beginning if the file doesn't exist

        # Define columns based on benchmark type
        if benchmark_type == "QA":
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[1]
            columns = ['Question', 'Correct Answer', 'Predicted Answer']

        elif benchmark_type == "ContextQA":
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            context_col = 'context' if 'context' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]
            columns = ['Question', 'Context', 'Correct Answer', 'Predicted Answer']

        elif benchmark_type == "Arzuman":
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            options_col = 'options' if 'options' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]
            columns = ['Question', 'Correct Answer', 'Predicted Option']

        elif benchmark_type == "Reshad":
            # question_col = 'text' if 'text' in df.columns else df.columns[0] # Reshad's old eng. topic names version
            question_col = 'question' if 'question' in df.columns else df.columns[0] # Reshad's updated aze. version
            options_col = 'options' if 'options' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]
            columns = ['Question', 'Correct Answer', 'Predicted Topic']

        elif benchmark_type == "ARC":
            question_col = 'Azerbaijani_q' if 'Azerbaijani_q' in df.columns else df.columns[0]
            choices_col = 'choices' if 'choices' in df.columns else df.columns[1]
            answer_col = 'answerKey' if 'answerKey' in df.columns else df.columns[2]
            columns = ['Question', 'Correct Answer', 'Predicted Option']

        elif benchmark_type == "kMC":
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            choices_col = 'choices' if 'choices' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]
            columns = ['Question', 'Correct Answer', 'Predicted Option']

        elif benchmark_type == "mMC":
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            options_col = 'options' if 'options' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]
            columns = ['Question', 'Correct Answer', 'Predicted Option']

        elif benchmark_type == "QMC":
            context_col = 'context' if 'context' in df.columns else df.columns[1]
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            options_col = 'options' if 'options' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]
            columns = ['Context', 'Question', 'Correct Answer', 'Predicted Option']

        # Process predictions, skipping already processed rows
        for index, row in df.iterrows():
            if index < start_index:  # Skip already processed rows
                continue
            
            if benchmark_type == "QA":
                question = row[question_col]
                predicted_answer = handle_qa_prediction(question, model_name, mode)
                predictions.append([question, row[answer_col], predicted_answer])

            elif benchmark_type == "ContextQA":
                question = row[question_col]
                context = row[context_col]
                predicted_answer = handle_context_qa_prediction(question, context, model_name, mode)
                predictions.append([question, context, row[answer_col], predicted_answer])

            elif benchmark_type == "Arzuman":
                question = row[question_col]
                options = row[options_col]
                predicted_option = handle_multiple_choice_prediction(question, options, model_name, mode)
                predictions.append([question, row[answer_col], predicted_option])

            elif benchmark_type == "Reshad":
                question = row[question_col]
                options = row[options_col]
                predicted_topic = handle_topic_classification_prediction(question, options, model_name, mode)
                predictions.append([question, row[answer_col], predicted_topic])

            elif benchmark_type == "ARC":
                question = row[question_col]
                options_txt = row[choices_col]
                array = pd.array
                options_dict = eval(options_txt)
                options = options_dict['az_choices'].tolist()
                predicted_option = handle_arc_prediction(question, options, model_name, mode)
                predictions.append([question, row[answer_col], predicted_option])

            elif benchmark_type == 'kMC':
                question = row[question_col]
                options = row[choices_col]
                predicted_topic = handle_mmlu_aze_prediction(question, options, model_name, mode, ending, file)
                predictions.append([question, row[answer_col], predicted_topic])

            elif benchmark_type == 'mMC':
                question = row[question_col]
                options = row[options_col]
                predicted_topic = handle_gsm8k_prediction(question, options, model_name, mode)
                predictions.append([question, row[answer_col], predicted_topic])

            elif benchmark_type == 'QMC':
                question = row[question_col]
                context = row[context_col]
                options = row[options_col]
                predicted_topic = handle_quad_prediction(question, context, options, model_name, mode)
                predictions.append([question, context, row[answer_col], predicted_topic])

        # Convert predictions to DataFrame and save to Excel
        output_df = pd.DataFrame(predictions, columns=columns)
        # output_df.to_excel(output_filename, header=start_index == 0)  # Add headers only for the first write # v1
        output_df.to_excel(output_filename)  # Add headers for all cases # v2
        print(f"Predictions saved to {output_filename}")

    except Exception as e:
        print(f"Error occurred while storing predictions: {str(e)}")
        traceback.print_exc() 
        # Save partial results if any errors occur
        if predictions:
            output_df = pd.DataFrame(predictions, columns=columns)
            output_filename_partial = f"{benchmark_type}_{model_name}_predictions_partial.xlsx"
            output_df.to_excel(output_filename_partial)  # Save partial results with index
            print(f"Partial predictions saved to {output_filename_partial} due to error.")

# Main function to run benchmarks and store predictions with error handling
def run_benchmark_store_answers(model_name, benchmark_type, df, mode, ending, file):
    try:
        print('store prediction started')
        store_predictions(df, benchmark_type, model_name, mode, ending, file)
    except Exception as e:
        print(f"Error while running benchmark for {benchmark_type} with model {model_name}: {str(e)}")
        traceback.print_exc()

