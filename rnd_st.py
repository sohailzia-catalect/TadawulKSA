from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import PyPDF2
from langchain.llms import Together
import dotenv
import streamlit as st
import tiktoken
from openai import OpenAI
import numpy as np
import pandas as pd
import together

dotenv.load_dotenv(".env")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="SaudiXchange", page_icon=':shark:')

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# "mistralai/Mixtral-8x7B-Instruct-v0.1"

# "mistralai/Mistral-7B-Instruct-v0.2"

# togethercomputer/llama-2-70b-chat

paraphrase_llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.1,
    max_tokens=128,
    top_k=0.5,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

rag_llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.1,
    max_tokens=8196,
    top_k=0.5,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)


@st.cache_data
def get_pkl_df():
    pkl_file_path = r"C:\Users\Catalect\PycharmProjects\tadawulchat\data.pkl"
    df = pd.read_pickle(pkl_file_path)
    return df


def rephrase_question(user_question):
    prompt = ("You are given a question related to Saudi Exhcange. Rephrase it accordingly. "
              "The questions is : " + user_question + ". Rephrased question: ")
    return paraphrase_llm(prompt)


def get_contextual_answer(user_question, context):
    template = (
        f"You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you from website. "
        f"\n Make sure to be helpful and only consider the information provided to you. "
        f" \n Where possible, also present the information in a nice format. "
        f"\n If you think the question is irrelevant, just ask the user to go away!"
        "---------------------\n"
        f"{context}"
        "\n---------------------\n"
        f"Given this information, please answer the question: {user_question}. \nAnswer: ")

    return rag_llm(template)


user_question = st.text_input("Enter your question: ")
if user_question:
    response_output = client.embeddings.create(input=truncate_text_tokens(user_question),
                                               model="text-embedding-ada-002")
    paraphrased_question = rephrase_question(user_question)
    st.write("The paraphrased question: " + paraphrased_question)
    df = get_pkl_df()
    embedding = response_output.data[0].embedding
    df["Simscores"] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    results = df.sort_values("Simscores", ascending=False)
    st.write(results)
    context = results["Raw Content"].iloc[0]
    answer  = get_contextual_answer(paraphrased_question, context)
    st.write(answer)
