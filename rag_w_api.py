
# client = ollama.Client()


from base import *


def create_combined_prompt_context(context: str, question: str) -> str:
    """
    Create the prompt for the LLM to generate an answer based on the given context and question.
    """

    # v1
    # return f"""
    #     You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.

    #     **Example:**

    #     **Context:** Azərbaycan Respublikası Cənubi Qafqazda yerləşən bir ölkədir. İqtisadi və mədəni mərkəzi Bakı şəhəridir.
        
    #     **Question in Azerbaijani:** Azərbaycan Respublikasının paytaxtı haradır?

    #     **Generated Answer in Azerbaijani:** Bakı şəhəri.

    #     **Your Task:**

    #     **Context in Azerbaijani:** {context}

    #     **Question in Azerbaijani:** {question}

    #     Provide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.
    # """

    # v3 
    return (
        f"You are an answer generator AI in Azerbaijani. Your task is to generate concise answers based on the provided context and the given question.\n"
        f"Provide a clear and accurate answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
        "### CONTEXT ###\n\n"
        '"""\n\n'
        f"{context}\n\n"
        '"""\n\n'
        "### END CONTEXT ###\n\n"
        f"Question: {question} Answer in Azerbaijani:\n"
    )

#     return (
#         f"Sən Azərbaycan dilində cavablar yaradan süni intellektsən. Sənin vəzifən verilmiş kontekstə və təqdim olunan suallara əsaslanaraq qısa cavablar yaratmaqdır.\n"
#         f"Suala 1-2 cümlədən çox olmayan və 400 simvoldan az olan dəqiq və aydın cavab ver.\n\n"
# #        # f"Suala 1-2 cümlədən çox olmayaraq, ən çox 400 simvoldan ibarət olan dəqiq və aydın cavab ver.\n\n"
#         "### KONTEKST (CONTEXT) ###\n\n"
#         '"""\n\n'
#         f"{context}\n\n"
#         '"""\n\n'
#         "### KONTEKSTİN SONU  (CONTEXT END) ###\n\n"
#         f"Sual: {question} Cavab Azərbaycan dilində:\n"
#     )


    # v3.1 
    # return (
    #     f"You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.\n"
    #     f"Provide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.\n\n"
    #     "### CONTEXT ###\n\n"
    #     '"""\n\n'
    #     f"{context}\n\n"
    #     '"""\n\n'
    #     "### END CONTEXT ###\n\n"
    #     f"Question: {question} Answer in Azerbaijani:\n"
    # )
    

    # # v1.1
    # return (
    #     f"You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.\n"
    #     f"Provide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.\n\n"
    #     f"Context in Azerbaijani:\n"
    #     f"{context}\n\n"
    #     f"Question in Azerbaijani:\n"
    #     f"{question}\n\n"
    #     f"Answer:"
    # )

    # # v2.1
    # return (
    #     f"Context in Azerbaijani:\n"
    #     f"{context}\n\n"
    #     f"Question in Azerbaijani:\n"
    #     f"{question}\n\n"
    #     f"Answer:"
    # )



def get_answer_from_local_ollama_context(model: str, question: str, context: str, mode: str) -> str:
    """
    Send a prompt to the local Ollama model and retrieve the answer using the Ollama library.
    """
    global last_log_time

    # Combine context and question
    prompt = create_combined_prompt_context(context, question)

    # Prepare messages for v2
    system_message = "You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.\nProvide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.\n"
    messages = [
        # {'role': 'system', 'content': system_message},  # Set the system context # v2
        {'role': 'user', 'content': prompt}  # The user's prompt # v1, v2
    ]

    answer = ""

    try:
        if mode == 'local':
            # Use the local Ollama model to generate a response (no streaming)
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],  # v1
                # messages=messages,  # v2
                stream=False  # Disable streaming (changed from stream=True to stream=False)
            )

            answer = response['message']['content']


        elif mode == 'api':
            # Call OpenAI's GPT model to generate a response
            response = client_openai.chat.completions.create(
                model=model,
                messages=messages,
                # max_tokens=5,  # Limit tokens since the answer is just one letter
                temperature=0.1,  # Lower temperature for more deterministic answers
            )

            answer = response.choices[0].message.content

    except Exception as e:
        print(f"Error during processing: {e}")
        return "Error"

    # If the answer is too long, return a special message
    if len(answer) > 400:
        return 'Long answer'
    
    return answer.strip() if answer else "Error"


def get_evaluation_score_context(question: str, actual_answer: str, predicted_answer: str, benchmark_type: str, model_name: str) -> str:
    """
    Generate an evaluation score between 0 and 100 by comparing the actual and predicted answers.
    """

    prompt = f"""
            Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
            answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

            0-10: No answer or completely incorrect
            11-30: Significant errors or missing key information
            31-50: Some errors or incomplete information, but recognizable effort
            51-70: Mostly accurate with minor errors or omissions
            71-90: Very close to the actual answer with only minor discrepancies
            91-100: Accurate or nearly perfect match

            **Example:**

            **Question that asked in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
            **Actual Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.  
            **Predicted Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın ümumi aspektlərini öyrənir, mikroiqtisadiyyat isə fərdi bazarları təhlil edir.  
            **Score (0 to 100):** 65

            **Your Task:**

            **Question that asked in Azerbaijani:** {question}

            **Actual Answer in Azerbaijani:** {actual_answer}

            **Predicted Answer in Azerbaijani:** {predicted_answer}

            **Score (0 to 100):**
            """


    payload = {
        "model": MODEL_LLAMA_3_1_405B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }

    for attempt in range(NUM_RETRIES):
        try:
            completion = client_nvidia.chat.completions.create(**payload)
            if completion.choices:
                content = completion.choices[0].message.content
                if content:
                    score = content.strip()
                    save_evaluation_to_file(benchmark_type, model_name, actual_answer, predicted_answer, score)
                    return score
                logging.error("Content in response is None.")
            else:
                logging.error(f"Unexpected response format: {completion}")
        except Exception as e:
            logging.error(f"Request failed: {e}")
    return "Error"



def save_evaluation_to_file(benchmark_type: str, model_name: str, actual_answer: str, predicted_answer: str, score: str):
    """
    Save evaluation results to file.
    
    Parameters:
    - benchmark_type (str): Type of benchmark.
    - model_name (str): Name of the model.
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    - score (str): The evaluation score.
    """

    results = [[actual_answer, predicted_answer, score]]

    # Create the evaluation_scores directory if it doesn't exist
    evaluation_scores_dir = "evaluation_scores"
    os.makedirs(evaluation_scores_dir, exist_ok=True)

    df = pd.DataFrame(results, columns=["Actual Answer", "Predicted Answer", "LLM Score"])

    filename_csv = os.path.join(evaluation_scores_dir, f"{benchmark_type}_{model_name}_evaluation_scores.csv")


    # # Append to the CSV file if it exists; otherwise, create a new one # v1
    # df.to_csv(filename_csv, mode="a", header=not os.path.exists(filename_csv), index=False, encoding='utf-8-sig')


    if os.path.exists(filename_csv): # v2
        existing_df = pd.read_csv(filename_csv)

        if existing_df.empty:
            # If the file exists but is empty, write with header
            df.to_csv(filename_csv, mode="w", header=True, index=False, encoding='utf-8-sig')
        else:
            # If the file exists and has rows, append without header
            df.to_csv(filename_csv, mode="a", header=False, index=False, encoding='utf-8-sig')
    else:
        # If the file does not exist, create it with header
        df.to_csv(filename_csv, mode="w", header=True, index=False, encoding='utf-8-sig')

