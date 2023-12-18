from dto import ChatbotRequest
from samples import list_card
import requests
import time
import logging
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import SingleStoreDB
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from datetime import datetime
from dotenv import load_dotenv

# load .env
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.environ.get('OPENAI_API_KEY')

# directory path
CONFIG_DIR = os.path.dirname(__file__) + "/config" # data directory
TOPIC_LIST_TXT = os.path.join(CONFIG_DIR, "topic_list.txt")
HISTORY_DIR = os.path.dirname(__file__) + "/chat_histories" # data directory
PARSE_TOPIC_TEMPLATE = os.path.join(CONFIG_DIR, "parse_topic.txt")
VECTOR_SEARCH_TEMPLATE = os.path.join(CONFIG_DIR, "vector_search.txt")
GENERATE_PROMPT_TEMPLATE = os.path.join(CONFIG_DIR, "generate_prompt.txt")
RESPONSE_FORMULATION_TEMPLATE = os.path.join(CONFIG_DIR, "response_form.txt")

SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

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
def relavant_docs(table_name, query: str) -> list[str]:

    db = SingleStoreDB(
        OpenAIEmbeddings(),
        host=os.environ.get('DB_HOST'),
        port=os.environ.get('DB_PORT'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD'),
        database=os.environ.get('DB_NAME'),
        distance_strategy="DOT_PRODUCT",
        table_name=table_name
    )

    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(query, search_type='similarity', search_kwargs={"k":6})

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

### Chain ###
# create chain
def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )
# generate chain
def generate_chain(llm):
    chains = {
        # 질문에서 주제를 파악하는 체인
        "parse_topic_chain": create_chain(
            llm=llm,
            template_path=PARSE_TOPIC_TEMPLATE,
            output_key="topic",
        ),
        # 주제와 관련된 문서를 찾아서 prompt를 생성하는 체인
        "generate_prompt_chain": create_chain(
            llm=llm,
            template_path=GENERATE_PROMPT_TEMPLATE,
            output_key="prompt",
        ),
        # prompt를 입력받아 답변을 생성하는 체인
        "response_form_chain": create_chain(
            llm=llm,
            template_path=RESPONSE_FORMULATION_TEMPLATE,
            output_key="output",
        ),
        "default_chain": ConversationChain(llm=llm, output_key="text")
    }

    return chains

def callback_handler(request: ChatbotRequest) -> dict:
    user_message = request.userRequest.utterance

    conversation_id = "chat_" + datetime.now().strftime("%Y%m%d%H%M%S")
    history_file = load_conversation_history(conversation_id)

    # topic_list.txt 중에서 해당하는 주제를 찾아서 target_topic에 저장
    context = dict(user_message=user_message)
    context["topic_list"] = read_prompt_template(TOPIC_LIST_TXT)
    
    llm = ChatOpenAI(temperature=0.1, max_tokens=2048, model="gpt-3.5-turbo")
    chains = generate_chain(llm)

    # 해당 topic에 맞는 문서를 DB에서 찾아서 relavant documents을 search
    topic = chains["parse_topic_chain"].run(context)
    context["target_topic"] = topic
    if "채널" in topic:
        table_name = "kakao_channel"
    elif "싱크" in topic:
        table_name = "kakao_sync"
    elif "소셜" in topic:
        table_name = "kakao_social"
    else:
        table_name = "kakao_etc"
    print("Target topic: ", topic)

    # change code to user function call [질문]
    context["related_documents"] = relavant_docs(table_name, user_message)

    # related documents를 이용해 prompt를 생성
    prompt = dict(topic="", content="")
    #generate prompt using related documents
    generate_prompt_chain = chains["generate_prompt_chain"]
    prompt["topic"]=context["target_topic"]
    prompt["content"] = generate_prompt_chain.run(context)

    response_from_chain = chains["response_form_chain"]
    answer = response_from_chain.run(prompt)

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
