from dto import ChatbotRequest
from samples import list_card
import requests
import time
import logging
import openai
import os
import json
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import SingleStoreDB
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from datetime import datetime

# load .env
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.environ.get('OPENAI_API_KEY')

# OpenAI API 키 설정
#openai.api_key = os.environ.get('OPENAI_API_KEY')
HUGGINFACE_API_KEY = os.environ.get('HUGGINFACE_API_KEY')

CONFIG_DIR = os.path.dirname(__file__) + "/config" # data directory
HISTORY_DIR = os.path.dirname(__file__) + "/chat_histories" # data directory
LLM_COMPLETION_TEMPLATE = os.path.join(CONFIG_DIR, "s2_llm_completion.txt")

SYSTEM_MSG = "당신은 SingleStore에 대한 질문에 대답해주는 서비스 제공자입니다."

logger = logging.getLogger("Callback")

def get_conn_string():
    connection_user = os.environ.get('DB_USER')
    connection_password = os.environ.get('DB_PASSWORD')
    connection_host = os.environ.get('DB_HOST')
    connection_port = os.environ.get('DB_PORT')
    db_name = os.environ.get('DB_NAME')
    return f"singlestoredb://{connection_user}:{connection_password}@{connection_host}:{connection_port}/{db_name}"

os.environ["SINGLESTOREDB_URL"] = get_conn_string()

### Prompt ###
# read prompt template
def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

# get relevant documents from DB
def relavant_docs(query: str, retriever) -> list[str]:

    docs = retriever.get_relevant_documents(query, search_type='similarity', search_kwargs={"k":20})
    #docs = db.similarity_search(query)
    #print(docs)
    str_docs = [doc.page_content for doc in docs]

    return str_docs

### History ### 
def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)

def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)

def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)

def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )

    return memory.buffer

## Request ##
def send_request(url, method, headers, payload):
  response = requests.request(method, url, headers=headers, data=payload)
  return response.json()

def call_llm_completion(context, max_tokens=2048, temperature=0, stream="fasle", frequency_penalty=1.0):
    
    prompt_suffix = "<|endoftext|>\n# AI 인공지능:\n"
    prompt = "# The following section enclosed by ''' is an introduction to singlestore. Please use this as a guide to answer questions. Please keep your answers polite. And the answer should be in Korean\n\n"
    info_text = "''' SingleStore Materials : " + context["relavant_documents"] + "'''\n\n"
    prompt = prompt + info_text  + prompt_suffix
    #print(prompt)
    request_body = {
        "model": "maru-red-summ",
        "max_tokens": str(max_tokens),
        "temperature": str(temperature),
        "stream": "false",
        "frequency_penalty": str(frequency_penalty),
        "stop": [
        "# Human 사람"
    ],
    "prompt": prompt
    }
    result = send_request("https://endpoint-prod-demo.ai-hub.9rum.cc/v1/completions", "POST", {}, json.dumps(request_body))

    return result


def callback_handler(request: ChatbotRequest) -> dict:
    # setup context
    user_message = request.userRequest.utterance
    context = dict(user_message=user_message)

    # get conversation history
    conv_id = "chat_" + datetime.now().strftime("%Y%m%d%H%M%S")
    history_file = load_conversation_history(conv_id)

    # need to apply conversation history to context
    #context["chat_history"] = get_chat_history(conv_id)
    
    # Connect to DB
    table_name = "singlestore"
    db = SingleStoreDB(
        HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask"),
        host=os.environ.get('DB_HOST'),
        port=os.environ.get('DB_PORT'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD'),
        database=os.environ.get('DB_NAME'),
        distance_strategy="DOT_PRODUCT",
        table_name=table_name
    )
    retriever = db.as_retriever()
    
    # get relevant documents from DB
    relavant_documents = relavant_docs(user_message, retriever) 
    relavant_documents_str = '\n'.join(relavant_documents)
    context["relavant_documents"] = relavant_documents_str

    # get answer from LLM
    result = call_llm_completion(context, max_tokens=2048, temperature=0, stream="fasle", frequency_penalty=1.0)
    answer = result['choices'][0]['text'] if result['choices'] else "No Result"

    # save conversation history
    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }

    url = request.userRequest.callbackUrl
    if url:
        requests.post(url=url, json=payload)
