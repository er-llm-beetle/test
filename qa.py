
from base import *

# from datetime import datetime, timedelta
# import threading


# Initialize the Ollama client
# client = ollama.Client()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')





def create_combined_prompt(question: str) -> str:
    """
    Create the prompt for the LLM to generate an answer based on the question.
    """
    # v1.1
    # return (
    #     f"You are an AI designed to generate answers in Azerbaijani based on the following questions.\n"
    #     f"Please provide a clear and concise answer in 1-2 sentences.\n\n"
    #     f"Question in Azerbaijani:\n"
    #     f"{question}\n\n"
    #     f"Answer:"
    # )

    # 
    return (
        f"You are an AI designed to generate concise answers in Azerbaijani based on the following questions.\n"
        f"Provide a clear and accurate answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
        f"Question in Azerbaijani:\n"
        f"{question}\n\n"
        f"Answer in Azerbaijani:"
    )

# 
#    return (
#        f"Sən aşağıdakı suallara əsaslanaraq Azərbaycan dilində qısa cavablar vermək üçün hazırlanmış bir süni intellektsən.\n"
#        f"Suala 1-2 cümlədən çox olmayaraq, ən çox 400 simvoldan ibarət olan dəqiq və aydın cavab ver.\n\n"
#        f"Sual:\n"
#        f"{question}\n\n"
#        f"Cavab Azərbaycan dilində:"
#    )



def get_answer_from_local_ollama(model: str, question: str) -> str:
    """
    Send a prompt to the local Ollama model and retrieve the answer using the ollama library.
    """

    prompt = create_combined_prompt(question)
    
    try:
        stream = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        
        answer = ''

        for chunk in stream:
            answer += chunk['message']['content']

    
    except Exception as e:
        logging.error(f"Request to local Ollama failed: {e}")
        # answer = "Error"
    

    if len(answer) > 400:
        return 'Long answer'
    return answer.strip() if answer else "Error"







def get_evaluation_score(question: str, actual_answer: str, predicted_answer: str) -> str:
    """
    Generate an evaluation score between 0 and 100 by comparing the actual and predicted answers.
    """

    # httpx_client = httpx.Client(http2=True, verify=False)

    # Initialize OpenAI client for NVIDIA
    # client = OpenAI(base_url=BASE_URL_LLM, api_key=API_KEY_LLM, http_client=httpx_client)


    # v2
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

            **Question that asked:** {question}

            **Actual Answer:** {actual_answer}

            **Predicted Answer:** {predicted_answer}

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
            completion = client_openai.chat.completions.create(**payload)
            if completion.choices:
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
                logging.error("Content in response is None.")
            else:
                logging.error(f"Unexpected response format: {completion}")
        except Exception as e:
            logging.error(f"Request failed: {e}")
    return "Error"



def calculate_bleu_score(actual_answer: str, predicted_answer: str) -> float:
    """
    Calculate BLEU score for the given actual and predicted answers.
    """
    reference = actual_answer.split()
    candidate = predicted_answer.split()
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
    # print('bleu:', 25 * bleu_score)
    return 25 * bleu_score

def calculate_rouge_score(actual_answer: str, predicted_answer: str) -> dict:
    """
    Calculate ROUGE score for the given actual and predicted answers.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(actual_answer, predicted_answer)

    normalized_scores = {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

    # print('rouge:', 25 * (0.33 * normalized_scores['rouge1'] + 0.33 * normalized_scores['rouge2'] + 0.33 * normalized_scores['rougeL']))
    return 25 * (0.33 * normalized_scores['rouge1'] + 0.33 * normalized_scores['rouge2'] + 0.33 * normalized_scores['rougeL'])


def calculate_levenshtein_score(actual_answer: str, predicted_answer: str) -> int:
    """
    Calculate Levenshtein distance between actual and predicted answers.
    """

    max_len = max(len(actual_answer), len(predicted_answer))

    if max_len == 0:  # both strings are empty
        return 1

    # print('levenshtein:', 25 * (1 - (Levenshtein.distance(actual_answer, predicted_answer) / max_len)))
    return 25 * (1 - (Levenshtein.distance(actual_answer, predicted_answer) / max_len))





def get_answer_from_local_ollama_v2(model: str, question: str) -> str:
    """
    Send a prompt to the local Ollama model and retrieve the answer using the ollama library.
    """
    prompt = create_combined_prompt(question)
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
        )

        answer = response['message']['content']
        # print("answer:", answer)
        
    except Exception as e:
        logging.error(f"Request to local Ollama failed: {e}")
        # answer = "Error"
    
    if len(answer) > 400:
        return 'Long answer'
    return answer.strip() if answer else "Error"

