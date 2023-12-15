import openai
import os
#import sqlalchemy
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SingleStoreDB
from dotenv import load_dotenv
from sqlalchemy import *

# load .env
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.environ.get('OPENAI_API_KEY')

def get_conn_string():
    connection_user = os.environ.get('DB_USER')
    connection_password = os.environ.get('DB_PASSWORD')
    connection_host = os.environ.get('DB_HOST')
    connection_port = os.environ.get('DB_PORT')
    db_name = os.environ.get('DB_NAME')
    return f"singlestoredb://{connection_user}:{connection_password}@{connection_host}:{connection_port}/{db_name}"

os.environ["SINGLESTOREDB_URL"] = get_conn_string()

# Loading 및 directory 설정
DATA_DIR = os.path.dirname(__file__) + "/data" # data directory
LOADER_DICT = {
    "txt": TextLoader
}
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"
EMBED_MODEL = "text-embedding-ada-002"
ids_id = 0

def upload_embedding_from_file(file_path, file_name):
    data = TextLoader(file_path).load()
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    doc_topic = file_name.replace(".txt","").split("_")[2]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 40)
    docs = text_splitter.split_documents(data)

    for doc in docs:
        doc.page_content = f"Topic:{doc_topic}\n{doc.page_content}" 
        #print(doc.page_content)

    if doc_topic == "카카오톡채널":
        table_name = "kakao_channel"
    elif doc_topic == "카카오싱크":
        table_name = "kakao_sync"
    elif doc_topic == "카카오소셜":
        table_name = "kakao_social"
    else:
        table_name = "kakao_etc"

    embeddings = OpenAIEmbeddings()

    SingleStoreDB.from_documents(
        docs,
        embeddings,
        table_name=table_name,
    )
    #print(file_name, ' Vectorization complete!')

def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                file_name = file.replace(".txt","").split("_")[2]
                #print(file_name)
                try:
                    upload_embedding_from_file(file_path, file_name)
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")
                    failed_upload_files.append(file_path)

def main():
    upload_embeddings_from_dir(DATA_DIR)

if __name__ == "__main__":
    main()