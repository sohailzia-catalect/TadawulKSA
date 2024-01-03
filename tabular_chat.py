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
from openai import OpenAI
import numpy as np
import pandas as pd
from langchain import PromptTemplate

dotenv.load_dotenv(".env")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="SaudiXchange", page_icon=':shark:')

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_docs():
    all_text = ""

    for folder in os.listdir(r"C:\Users\Catalect\PycharmProjects\tadawulchat\report_data"):
        pdf_reader = PyPDF2.PdfReader(os.path.join(r"C:\Users\Catalect\PycharmProjects\tadawulchat\report_data", folder))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        all_text += text

    return all_text

# loaded_text = load_docs()
#
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1024, chunk_overlap=0)
#
# docs = text_splitter.split_text(loaded_text)
#
# vectorstore = FAISS.from_texts(docs, embedding=embeddings)
# vectorstore.save_local("report_index_storage")
#

vectorstore = FAISS.load_local(r"C:\Users\Catalect\PycharmProjects\tadawulchat\report_index_storage", embeddings)

# "mistralai/Mixtral-8x7B-Instruct-v0.1"

def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1,
    max_tokens=512,
    top_k=0.5,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

@st.cache_data
def get_pkl_df():
    pkl_file_path = r"C:\Users\Catalect\PycharmProjects\tadawulchat\references.pkl"
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


retriever = vectorstore.as_retriever(k=1)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="refine", verbose=True)
# answer = qa.run("how do I become an investor?")

user_question = st.text_input("Enter your question:")
if user_question:
    answer = qa.run(user_question + ". Please make sure to be be concise and be a helpful chat assistant.  " )
    st.write("Answer:", answer)
    st.write("Ref: " + get_url(answer))
