import os
#import sqlalchemy
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import SingleStoreDB
from dotenv import load_dotenv
from sqlalchemy import *
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
#from sqlalchemy import create_engine

#https://www.youtube.com/watch?v=Kn7SX2Mx_Jk

# load .env
load_dotenv()

def get_conn_string():
    connection_user = os.environ.get('DB_USER')
    connection_password = os.environ.get('DB_PASSWORD')
    connection_host = os.environ.get('DB_HOST')
    connection_port = os.environ.get('DB_PORT')
    db_name = os.environ.get('DB_NAME')
    return f"singlestoredb://{connection_user}:{connection_password}@{connection_host}:{connection_port}/{db_name}"

os.environ["SINGLESTOREDB_URL"] = get_conn_string()
HUGGINFACE_API_KEY = os.environ.get('HUGGINFACE_API_KEY')

# Loading 및 directory 설정
DATA_DIR = os.path.dirname(__file__) + "/singlestore" # data directory

def upload_embedding_from_file(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    doc_topic = file_name.replace(".pdf","")

    # PDF 파일 로드
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
    except Exception as e:
        print(f"Error during loading PDF: {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 0)
    docs = text_splitter.split_documents(data)
    #print(docs[0].page_content)

    # 임베딩 모델 로드
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    #model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # SingleStoreDB 설정
    table_name = "singlestore"
    #model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    for doc in docs:
        doc.page_content = f"Topic:{doc_topic}\n{doc.page_content}"
    
    modelPath = "jhgan/ko-sroberta-multitask"
    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': True}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )    
    SingleStoreDB.from_documents(
        docs,
        embeddings,
        table_name=table_name,
    )
    #print(file_name, 'Uploading to VectorDB completed!')

def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []
    
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                #print(file_name)
                try:
                    upload_embedding_from_file(file_path)
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")
                    failed_upload_files.append(file_path)

def main():
#    connection_url = get_conn_string()
#    print(connection_url)
#    db = sqlalchemy.create_engine(connection_url)
#    conn = db.connect()
     upload_embeddings_from_dir(DATA_DIR)
#    conn.close()
#    db.dispose()

if __name__ == "__main__":
    main()
