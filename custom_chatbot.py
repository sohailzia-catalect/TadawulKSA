import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS
from llama_index.llms import OpenAILike

from langchain.llms import Together
import tiktoken
from openai import OpenAI as orig_openai
import numpy as np
import pandas as pd
import dotenv

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

dotenv.load_dotenv(".env")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

client = orig_openai(api_key=os.getenv("OPENAI_API_KEY"))

openai.api_key = os.getenv("OPENAI_API_KEY")

data_path = r"C:\Users\Catalect\PycharmProjects\tadawulchat\data"

template = (
    "You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you from website. "
    "\n Make sure to be helpful and only consider the information provided to you. "
    " \n Where possible, also present the information in a nice format. "
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question while providing as much details as possible: {query_str}\n"
)
qa_template = Prompt(template)


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_data():
    reader = SimpleDirectoryReader(input_dir=data_path)
    data = reader.load_data()
    return data


@st.cache_data
def get_rag_llm():
    rag_llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.01,
        max_tokens=8196,
        top_k=0.5,
        together_api_key=os.getenv("TOGETHER_API_KEY"))
    return rag_llm


# "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm = OpenAILike(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_base="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),
    is_chat_model=True,
    is_function_calling_model=False,
    temperature=0.1,
)


@st.cache_data
def get_query_engine():
    service_context = ServiceContext.from_defaults(
        llm=llm
    )
    # index = VectorStoreIndex.from_documents(get_data(), service_context=service_context)
    # index.storage_context.persist("faiss_index")
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir="faiss_index"))

    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)
    return query_engine


def get_response(question):
    query_engine = get_query_engine()

    response = query_engine.query(question)
    return response


@st.cache_data
def get_pkl_df():
    pkl_file_path = r"C:\Users\Catalect\PycharmProjects\tadawulchat\data.pkl"
    df = pd.read_pickle(pkl_file_path)
    return df


def get_contextual_answer(user_question, context):
    prompT = (
        f"You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you from website. "
        f"\n Make sure to be helpful and only consider the information provided to you. "
        f" \n Where possible, also present the information in a nice format. "
        f"\n If you think the question is irrelevant, just ask the user to go away!"
        "---------------------\n"
        f"{context}"
        "\n---------------------\n"
        f"Given this information, please answer the question: {user_question}. \nAnswer: ")
    return get_rag_llm()(prompT)


user_question = st.text_input("Enter your question: ")
if user_question:
    answer = get_response(user_question).response

    response_output = client.embeddings.create(input=truncate_text_tokens(answer),
                                               model="text-embedding-ada-002")
    df = get_pkl_df()
    embedding = response_output.data[0].embedding
    df["Simscores"] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    results = df.sort_values("Simscores", ascending=False)
    st.warning("was able find relevant data with conf score of " + str(results["Simscores"][0]))
    st.write(results)

    context = results["Raw Content"].iloc[0]
    answer = get_contextual_answer(user_question, context)

    st.write(answer)
    st.write("For more detailed information, go to: " + results["Reference"][0])
