import os
from typing import Optional, Any
from uuid import UUID

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import base64
from langchain.prompts import PromptTemplate
import dotenv
import pandas as pd
import tiktoken
import numpy as np
from openai import OpenAI

dotenv.load_dotenv(".env")

st.set_page_config(page_title="KSA XChange Agent", page_icon="🤖")
st.header("🤖 Saudi Exchange VAI Assistant", divider='rainbow')

openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

template = """You are Saudi Stock Exchange chat assistant and so act like one. 
Your job is to answer queries with information provided to 
you. Make sure to be helpful and only consider the information provided to you. Don't try to make up an answer. 
If you don't know the answer, simply say that you don't know.
Where possible, also present the information in a nice format. "{context}" Given above information, please answer the 
question while providing as much details as possible and if you cannot find answer to the question, 
simple say that you do not know. Question: {question}
Answer: 
"""


def load_docs(files):
    print("In progress: Loading data into a single text.")
    all_text = ""
    for file_path in os.listdir(files):
        file_extension = file_path.split('/')[-1].split('.')[-1]
        file_path = "data/" + str(file_path)
        if file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == "txt":
            with open(file_path, 'r') as file:
                file_contents = file.read()
            all_text += file_contents
    print("Done: Loading data into a single text.")
    return all_text


filename = r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data"

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

prompt = PromptTemplate(
    template=template,
    input_variables=[
        'context',
        'question',
    ]
)


def split_texts(text, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)

    return splits


main_bg = "images/darken_image.jpg"
main_bg_ext = ".jpg"


def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64(main_bg)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/jpg;base64,{img}");
background-size: 100%;
background-position: center;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/jpg;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}


[data-testid="stToolbar"] {{
right: 2rem;

}}
.stTextInput {{
      position: fixed;
      bottom: 3rem;
      background: rgba(100,0,0,0);
    }}

</style>

"""

st.markdown(page_bg_img, unsafe_allow_html=True)

hide_st_style = """
 <style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
 </style>

"""

st.markdown(hide_st_style, unsafe_allow_html=True)


def get_reference(context):
    response_output = client.embeddings.create(input=truncate_text_tokens(context),
                                               model="text-embedding-ada-002")

    df = get_pkl_df()
    embedding = response_output.data[0].embedding
    df["Simscores"] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    results = df.sort_values("Simscores", ascending=False)
    max_conf_score = results.reset_index()["Simscores"][0]
    print("max_conf_score: ", max_conf_score)
    if max_conf_score >= 0.78:
        return context + "\n\nFor more information, go to: " + results.reset_index()["Reference"][0]
    return context


@st.cache_resource()
def configure_retriever():
    loaded_text = load_docs(filename)

    # Split the document into chunks
    splits = split_texts(loaded_text, chunk_size=1500,
                         overlap=200)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings()

    # Define retriever
    vectorstore = FAISS.from_texts(splits, embeddings)
    retriever = vectorstore.as_retriever(k=5)

    return retriever


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@st.cache_data
def get_pkl_df():
    pkl_file_path = r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data_updated.pkl"
    df = pd.read_pickle(pkl_file_path)
    return df


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        self.text = get_reference(self.text)
        self.container.markdown(self.text)


@st.cache_resource()
def get_qa_chain():
    retriever = configure_retriever()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k", temperature=0.4, streaming=True, api_key=os.getenv("OPENAI_API_KEY"))

    qa_chain = RetrievalQA.from_llm(llm, retriever=retriever, verbose=True)
    return qa_chain


# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
avatar_emoji = {"human": "👳", "ai": "🤖"}

for msg in msgs.messages:
    st.chat_message(avatars[msg.type], avatar=avatar_emoji[msg.type]).write(msg.content)

if user_query := st.text_input(label="Ask me stuff", label_visibility="collapsed"):
    print("-----------")
    print("Question asked: ", user_query)
    st.chat_message("user", avatar="👳").write(user_query)
    msgs.add_user_message(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        stream_handler = StreamHandler(st.empty())
        response = get_qa_chain().run(user_query, callbacks=[stream_handler])
        print("Answer generated: ", response)
        msgs.add_ai_message(response)
