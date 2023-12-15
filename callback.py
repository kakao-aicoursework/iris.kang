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
from dotenv import load_dotenv

# load .env
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.environ.get('OPENAI_API_KEY')

CONFIG_DIR = os.path.dirname(__file__) + "/config" # data directory
TOPIC_LIST_TXT = os.path.join(CONFIG_DIR, "topic_list.txt")
PARSE_TOPIC_TEMPLATE = os.path.join(CONFIG_DIR, "parse_topic.txt")
VECTOR_SEARCH_TEMPLATE = os.path.join(CONFIG_DIR, "vector_search.txt")
GENERATE_PROMPT_TEMPLATE = os.path.join(CONFIG_DIR, "generate_prompt.txt")
#RESPONSE_FORMULATION_TEMPLATE = os.path.join(CONFIG_DIR, "response_form.txt")

SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."

logger = logging.getLogger("Callback")

def get_conn_string():
    connection_user = os.environ.get('DB_USER')
    connection_password = os.environ.get('DB_PASSWORD')
    connection_host = os.environ.get('DB_HOST')
    connection_port = os.environ.get('DB_PORT')
    db_name = os.environ.get('DB_NAME')
    return f"singlestoredb://{connection_user}:{connection_password}@{connection_host}:{connection_port}/{db_name}"

os.environ["SINGLESTOREDB_URL"] = get_conn_string()

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

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
    docs = retriever.get_relevant_documents(query, search_kwargs={"k": 50})

    str_docs = [doc.page_content for doc in docs]
    return str_docs

def generate_chain(llm):
    chains = {
        "parse_topic_chain": create_chain(
            llm=llm,
            template_path=PARSE_TOPIC_TEMPLATE,
            output_key="topic",
        ),
        "generate_prompt_chain": create_chain(
            llm=llm,
            template_path=GENERATE_PROMPT_TEMPLATE,
            output_key="prompt",
        ),
        '''
        "response_form_chain": create_chain(
            llm=llm,
            template_path=RESPONSE_FORMULATION_TEMPLATE,
            output_key="output",
        ),
        '''
        "default_chain": ConversationChain(llm=llm, output_key="text")
    }

    return chains

def callback_handler(request: ChatbotRequest) -> dict:
    user_message = request.userRequest.utterance
    print(user_message)
    context = dict(user_message=user_message)
    context["topic_list"] = read_prompt_template(TOPIC_LIST_TXT)
    
    llm = ChatOpenAI(temperature=0.1, max_tokens=2048, model="gpt-3.5-turbo")

    chains = generate_chain(llm)
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

    context["related_documents"] = relavant_docs(table_name, user_message)

    #generate prompt using related documents
    generate_prompt_chain = chains["generate_prompt_chain"]
    output_text = generate_prompt_chain.run(context)
    print(output_text)
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }

    url = request.userRequest.callbackUrl
    if url:
        requests.post(url=url, json=payload)
