import pandas as pd
import openai
import chromadb
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
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
    
def send_message(message_log, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature
    )

    return response.choices[0].message.content

def main():
    chat_source = []
    df = read_data(file_path)
    char_source = add2vectordb(df)

    message_log = [
        {
            "role": "system",
            "content": '''
            당신은 고객의 질문에 대답하는 챗봇입니다.\
            제안된 답변은 신뢰할 수 있고, 관심 있는 주제에 기반해야 합니다.\
            답변을 위해 다음의 정보를 이용해주세요. \
            카카오톡 채널 정보: {chat_source}
            고객에게 친절하고 상세한 답변을 제공하는 챗봇이 되어주세요.
            '''
        }
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()
if __name__ == "__main__":
    main()