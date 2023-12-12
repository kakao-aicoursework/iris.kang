import pandas as pd
import openai
import os
import re
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


# OpenAI API 키 설정
openai_key = os.environ["OPENAI_API_KEY"]
file_path = './project_data_카카오싱크.txt'
template_path = './chatbot_prompt.txt'

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    sections = content.split('#')[1:]  # Split the file content into sections
    parsed_data = {}
    for section in sections:
        title, content = section.strip().split('\n', 1)
        parsed_data[title] = content.strip()

    return parsed_data

def create_chain(llm, prompt_template, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=prompt_template,
        ),
        output_key=output_key,
        verbose=True,
    )

def create_chatbot(file_path, question, response=""):

    sink_data = parse_file(file_path)

    chat_model = ChatOpenAI(temperature=0.1, max_tokens=500, model='gpt-3.5-turbo')
    chains = []
    output_keys = []
    for index, (title, content) in enumerate(sink_data.items()):
        template = read_prompt_template(template_path)
        prompt_template = template.replace("{title}", title).replace("{content}", content)
        chain = create_chain(chat_model, prompt_template, f'output_{index}')
        chains.append(chain)
        output_keys.append(f'output_{index}')
    # 
    #final_output_key = chains[-1].output_key
    #final_output_key = f'output_{len(chains) - 1}'

    chatbot_chain = SequentialChain(
        chains=chains,
        input_variables=["question", "response"],
        output_variables=output_keys,         
        verbose=True,
    )
    context = dict(
        question=question,
        response=response
    )
    context = chatbot_chain(context)

    context["answers"] =[]
    context = chains[-1](context)
    context["answers"].append(context[f"output_{len(chains) - 1}"])
    context["answers"] = context["answers"][0]

    #def get_response(question, response=""):
    #    context = {
    #      "question": question,
    #      "response": response
    #    }
    #    context = chatbot_chain(context)
    #    return context
    
    return context["answers"]

# Usage
question="카카오싱크 기능이 무엇이 있는지 소개해주세요"
run_chatbot = create_chatbot(file_path, question, "")
print(run_chatbot)
