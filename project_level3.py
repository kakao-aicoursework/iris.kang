
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
GENERATE_PROMPT_TEMPLATE = os.path.join(CONFIG_DIR, "generate_prompt.txt")
RESPONSE_FORMULATION_TEMPLATE = os.path.join(CONFIG_DIR, "response_form.txt")

def get_conn_string():
    connection_user = os.environ.get('DB_USER')
    connection_password = os.environ.get('DB_PASSWORD')
    connection_host = os.environ.get('DB_HOST')
    connection_port = os.environ.get('DB_PORT')
    db_name = os.environ.get('DB_NAME')
    return f"singlestoredb://{connection_user}:{connection_password}@{connection_host}:{connection_port}/{db_name}"

os.environ["SINGLESTOREDB_URL"] = get_conn_string()
#https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846

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
    docs = retriever.get_relevant_documents(query, search_type='similarity', search_kwargs={"k":0.5})

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
        "response_form_chain": create_chain(
            llm=llm,
            template_path=RESPONSE_FORMULATION_TEMPLATE,
            output_key="output",
        ),
        "default_chain": ConversationChain(llm=llm, output_key="text")
    }

    return chains

def generate_answer(user_message) -> dict[str, str]:
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
    print("Target topic: ", topic)
    context["related_documents"] = relavant_docs(table_name, user_message)
    #print(context["related_documents"])

    prompt = dict(topic="", content="")
    #generate prompt using related documents
    generate_prompt_chain = chains["generate_prompt_chain"]
    prompt["topic"]=context["target_topic"]
    prompt["content"] = generate_prompt_chain.run(context)
    print("Prompt : ", prompt["content"])

    response_from_chain = chains["response_form_chain"]
    answer = response_from_chain.run(prompt)

    return answer

def main():
    print(generate_answer('카카오톡채널을 소개하고 사용 방법을 알려주세요'))

if __name__ == "__main__":
    main()