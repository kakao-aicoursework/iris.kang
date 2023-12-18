
import os
import requests
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import SingleStoreDB
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from datetime import datetime

# load .env
load_dotenv()

# OpenAI API 키 설정
#openai.api_key = os.environ.get('OPENAI_API_KEY')
HUGGINFACE_API_KEY = os.environ.get('HUGGINFACE_API_KEY')

# directory path
CONFIG_DIR = os.path.dirname(__file__) + "/config" # data directory
HISTORY_DIR = os.path.dirname(__file__) + "/chat_histories" # data directory
LLM_COMPLETION_TEMPLATE = os.path.join(CONFIG_DIR, "s2_llm_completion.txt")

### Vector Database ###
# DB connection string from .env
def get_conn_string():
    connection_user = os.environ.get('DB_USER')
    connection_password = os.environ.get('DB_PASSWORD')
    connection_host = os.environ.get('DB_HOST')
    connection_port = os.environ.get('DB_PORT')
    db_name = os.environ.get('DB_NAME')
    return f"singlestoredb://{connection_user}:{connection_password}@{connection_host}:{connection_port}/{db_name}"

# get relevant documents from DB
def relavant_docs(query: str, retriever) -> list[str]:

    docs = retriever.get_relevant_documents(query, search_type='similarity', search_kwargs={"k":20})
    #docs = db.similarity_search(query)
    #print(docs)
    str_docs = [doc.page_content for doc in docs]

    return str_docs

# set DB connection string to environment variable
os.environ["SINGLESTOREDB_URL"] = get_conn_string()
#https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846

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

### Prompt ###
# read prompt template
def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

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

## Request ##
def send_request(url, method, headers, payload):
  response = requests.request(method, url, headers=headers, data=payload)
  return response.json()

# Generate Answer   
def generate_answer(user_message, conversation_id: str="default", db=None) -> dict[str, str]:
    retriever = db.as_retriever()
    #history_file = load_conversation_history(conversation_id)

    context = dict(user_message=user_message)

    relavant_documents = relavant_docs(user_message, retriever) 
    relavant_documents_str = '\n'.join(relavant_documents)
    context["relavant_documents"] = relavant_documents_str

    result = call_llm_completion(context, max_tokens=2048, temperature=0, stream="fasle", frequency_penalty=1.0)
    answer = result['choices'][0]['text'] if result['choices'] else "No Result"

    #log_user_message(history_file, user_message)
    #log_bot_message(history_file, answer)
    #print(result)

    return answer

### Main ###
def main():
    conv_id = "chat_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
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

    #print(generate_answer('SingleStore의 CPU 리소스 제한에 대해 알려줄래?', conv_id, db))
    print(generate_answer('SingleStore의 공용 클러스터의 Child Group에 대해 설명해줄래?', conv_id, db))
if __name__ == "__main__":
    main()