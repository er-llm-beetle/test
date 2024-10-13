import time
import httpx
from openai import OpenAI
import ollama
import pandas as pd
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein
from openai import OpenAI
import httpx
import os
from dotenv import load_dotenv
import re

load_dotenv()

# Configuration
BASE_URL_LLM = "https://integrate.api.nvidia.com/v1"
MODEL_LLAMA_3_1_405B = "meta/llama-3.1-405b-instruct"
MODEL_LLAMA_3_1_8B = "meta/llama-3.1-8b-instruct"
API_KEY_LLM = os.getenv('API_KEY_NVIDIA_LLM')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
NUM_RETRIES = 2
BASE_SLEEP_TIME = 1


# Basic logging config.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# SSL certificate problem fixing
httpx_client = httpx.Client(http2=True, verify=False)


# Version 1:
# Initialize OpenAI client for NVIDIA 405b
client_nvidia = OpenAI(
    base_url=BASE_URL_LLM, 
    api_key=API_KEY_LLM, 
    http_client=httpx_client
)

# Initialize client for GPT 
client_openai = OpenAI(
    # base_url=BASE_URL_LLM, 
    api_key=OPENAI_API_KEY, 
    http_client=httpx_client
)

# Initialize the Ollama client
client_ollama = ollama.Client()


# Version 2:
# initialize all clients in one line
# clients = {
#     'nvidia': OpenAI(base_url=BASE_URL_LLM, api_key=API_KEY_LLM, http_client=httpx_client),
#     'openai': OpenAI(api_key=OPENAI_API_KEY, http_client=httpx_client),
#     'ollama': ollama.Client(),
# }

# # and then acccess them
# client_nvidia = clients['nvidia']
# client_openai = clients['openai']
# client_ollama = clients['ollama']



# __all__ = ['ollama', 'openai'] # Only ollama and openai will be imported with  ->  from . import *
