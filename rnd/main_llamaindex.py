from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import PyPDF2
from langchain.llms import Together
from langchain.chains import RetrievalQA
import dotenv
import streamlit as st
import tiktoken
from openai import OpenAI as orig_openai
import numpy as np
import pandas as pd
from langchain import PromptTemplate

import openai
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    StorageContext,
    load_index_from_storage,
    Prompt,
    ServiceContext
)

from llama_index.llms import OpenAI

dotenv.load_dotenv("../.env")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="SaudiXchange", page_icon=':shark:')

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

client = orig_openai(api_key=os.getenv("OPENAI_API_KEY"))


def load_docs():
    all_text = ""

    for folder in os.listdir(r"/data"):

        pdf_reader = PyPDF2.PdfReader(os.path.join(r"/data", folder))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        all_text += text

    return all_text


loaded_text = load_docs()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=4096, chunk_overlap=0)
#
# docs = text_splitter.split_text(loaded_text)
#
# vectorstore = FAISS.from_texts(docs, embedding=embeddings)
# vectorstore.save_local("llama_index_storage")

vectorstore = FAISS.load_local(r"C:\Users\Catalect\PycharmProjects\tadawulchat\llama_index_storage", embeddings)


# "mistralai/Mixtral-8x7B-Instruct-v0.1"

def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


from llama_index.llms import OpenAILike

llm = OpenAILike(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_base="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),
    is_chat_model=True,
    is_function_calling_model=False,
    temperature=0.1,
)


@st.cache_data
def get_pkl_df():
    pkl_file_path = r"/rnd/references.pkl"
    df = pd.read_pickle(pkl_file_path)
    return df


def get_url(content):
    response_output = client.embeddings.create(input=truncate_text_tokens(content), model="text-embedding-ada-002")
    df = get_pkl_df()
    embedding = response_output.data[0].embedding
    df["Simscores"] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    results = df.sort_values("Simscores", ascending=False)
    st.write(df.Simscores[0])
    # if df.Simscores[0] < 0.7:
    #     return ""
    url = results.Ref.values[0]

    return url


dotenv.load_dotenv('../../.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

data_path = r"/data"

filename = r"/data"

template = (
    "You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you from website. "
    "\n Make sure to be helpful and only consider the information provided to you. "
    " \n Where possible, also present the information in a nice format. "
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}. "
    "\n If you think the question is irrelevant, just ask the user to go away!"
)
qa_template = Prompt(template)


@st.cache_data
def get_data(uploaded_files):
    # TODO should read from uploaded; just like how it's being done for langchain demo.
    reader = SimpleDirectoryReader(input_dir=data_path)
    data = reader.load_data()
    print(data)
    return data


@st.cache_resource
def get_query_engine():
    service_context = ServiceContext.from_defaults(llm=llm)

    index = VectorStoreIndex.from_documents(get_data(filename), service_context=service_context)

    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=1)
    return query_engine


def get_response(question):
    query_engine = get_query_engine()

    response = query_engine.query(question)
    return response


user_question = st.text_input("Enter your question:")
if user_question:
    answer = get_response(user_question)
    st.write("Answer:", answer.response)
    st.write("Ref: " + get_url(str(answer.response)))
