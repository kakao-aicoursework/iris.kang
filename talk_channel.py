import pandas as pd
import openai
import requests
import json
import chromadb
import os
import re

# OpenAI API 키 설정
openai_key = os.environ["OPENAI_API_KEY"]
file_path = './project_data_카카오톡채널.txt'

def read_data(file_path):    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r'^(#[^\n]+)((?:\n(?!#).*)*)'
    matches = re.findall(pattern, content, re.MULTILINE)

    section_titles = []
    section_contents = []

    for match in matches:
        title, content = match
        section_titles.append(title.strip('#').strip())
        section_contents.append(content.strip())

    data = pd.DataFrame({
        'Title': section_titles,
        'Content': section_contents
    })
    return data

def add2vectordb(df):
    client = chromadb.PersistentClient()

    collection = client.get_or_create_collection(
        name="kakaochannel_collection",
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    documents = []

    for idx in range(len(df)):
        item = df.iloc[idx]
        id = item['Title'].lower().replace(' ','-')
        document = f"{item['Title']}: {str(item['Content']).strip().lower()}"
        
        ids.append(id)
        documents.append(document)

    # DB 저장
    collection.add(
        documents=documents,
        ids=ids
    )
    # DB 쿼리 
    query_result = collection.query(query_texts=["기능"], n_results=1)
    return query_result

def kakaochannel_chatbot(chat_source, temperature=1, max_tokens=1024):
    prompt = f"""
카카오 채널 API에 대해 질문에 답해주세요.
제가 알고 싶은 카카오톡 채널에 대한 정보는 다음과 같습니다.\
다음의 정보를 이용해 카카오톡 채널에 대해 질문에 답해주세요. \
카카오톡 채널 정보: {chat_source}
"""
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
               "role": "system",
               "content" : "당신은 고객의 질문에 대답하는 챗봇입니다.\
                제안된 답변은 신뢰할 수 있고, 관심 있는 주제에 기반해야 합니다.\
                고객에게 친절하고 상세한 답변을 제공하는 챗봇이 되어주세요."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content

def main() :
    df = read_data(file_path)
    query_result = add2vectordb(df)
    print(kakaochannel_chatbot(query_result['documents'][0]))

if __name__ == "__main__":
    main()
