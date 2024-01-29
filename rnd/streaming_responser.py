from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import streamlit as st
import openai
import os

os.environ["OPENAI_API_KEY"] = "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"


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


query = st.text_input("input your query", value="Tell me a joke")
ask_button = st.button("ask")

# here is the key, setup a empty container first
chat_box = st.empty()
stream_handler = StreamHandler(chat_box)
chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", streaming=True, callbacks=[stream_handler])


def get_context():
    with open("../company_info_context.txt", 'r') as file:
        # Read the entire contents of the file
        file_contents = file.read()

    return file_contents


context = get_context()
prompt = f"You are Saudi Stock Exchange chat assistant. Your job is to provide information related to companies using the given context. Make sure to be helpful and only consider the information provided to you: {context}. \n Given this information, please answer: {query}"

if query and ask_button:
    response = chat([HumanMessage(content=prompt)])
    llm_response = response.content
