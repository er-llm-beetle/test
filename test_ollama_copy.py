import ollama
import logging
# import subprocess



def get_answer_from_local_ollama(model: str, question: str) -> str:
    """
    Send a prompt to the local Ollama model and retrieve the answer using the ollama library.
    """

    # Version 1:
    prompt = (  # v1
        f"You are an AI designed to generate concise answers in Azerbaijani based on the following questions.\n"
        f"Provide a clear answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
        f"Question in Azerbaijani:\n"
        f"{question}\n\n"
        f"Answer in Azerbaijani:"
    )

    # prompt = ( # v1_aze
    #    f"Sən aşağıdakı suallara əsaslanaraq Azərbaycan dilində qısa cavablar vermək üçün hazırlanmış bir süni intellektsən.\n"
    #    f"Suala 1-2 cümlədən çox olmayaraq, ən çox 400 simvoldan ibarət olan dəqiq və aydın cavab ver.\n\n"
    #    f"Sual:\n"
    #    f"{question}\n\n"
    #    f"Cavab Azərbaycan dilində:"
    # )


    # Version 2:
    # system_prompt = ( # v2
    #     f"You are an AI designed to generate concise answers in Azerbaijani based on the following questions.\n"
    #     f"Provide a clear answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
    # )
    # prompt_content = ( # v2
    #     f"Question in Azerbaijani:\n"
    #     f"{question}\n\n"
    #     f"Answer in Azerbaijani:"
    # )

    # messages = [ # v2 
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": prompt_content}
    # ]



    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}], # v1
            # messages=messages, # v2
        )

        answer = response['message']['content']
        # print("answer:", answer)
        
    except Exception as e:
        logging.error(f"Request to local Ollama failed: {e}")
        # answer = "Error"
    
    if len(answer) > 400:
        return 'Long answer'
    
    return answer.strip() if answer else "Error"



if __name__ == "__main__":
    model_name = "llama3.1_az_v2" 
    question = 'Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?'
    response = get_answer_from_local_ollama(model_name, question)
    print(f"Response: {response}")
