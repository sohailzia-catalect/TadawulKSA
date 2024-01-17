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
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import UnstructuredMarkdownLoader, UnstructuredURLLoader

dotenv.load_dotenv(".env")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="SaudiXchange", page_icon=':shark:')

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_docs():
    all_text = ""

    for folder in os.listdir(r"C:\Users\Catalect\PycharmProjects\tadawulchat\tabular_data"):
        pdf_reader = PyPDF2.PdfReader(
            os.path.join(r"C:\Users\Catalect\PycharmProjects\tadawulchat\tabular_data", folder))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        all_text += text

    return all_text


loaded_text = load_docs()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=0)

docs = text_splitter.split_text(loaded_text)

bm25_retriever = BM25Retriever.from_texts(docs)
bm25_retriever.k = 2

# vectorstore = FAISS.from_texts(docs, embedding=embeddings)
# vectorstore.save_local("index/report_index_storage")

vectorstore = FAISS.load_local(r"C:\Users\Catalect\PycharmProjects\tadawulchat\index\report_index_storage", embeddings)

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorstore.as_retriever()], weights=[0.7, 0.3]
)


# "mistralai/Mixtral-8x7B-Instruct-v0.1"
# "mistralai/Mistral-7B-Instruct-v0.2"
def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.1,
    max_tokens=512,
    top_k=0.5,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv("GEMINI_API_KEY"),
                             temperature=0.1,convert_system_message_to_human=True).client

retriever = vectorstore.as_retriever(k=1)

template = """You are a stock exchange analyst for Saudi Exchange. Use the following context to answer a question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible and don't make up own facts. Think properly before you reply. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Run chain

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=ensemble_retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# qa = RetrievalQA.from_chain_type(llm=llm, retriever=ensemble_retriever, chain_type="refine", verbose=True)
# answer = qa.run("how do I become an investor?")

user_question = st.text_input("Enter your question:")
if user_question:
    # answer = qa_chain.run(user_question)
    result = qa_chain({"query": user_question})
    st.write("Answer:", result["result"])

    result = qa_chain({"query": f"Your job is to reformat given data into nice looking format in markdown, only if it can be cultivated. Input:{result}.\n Reformated Markdown Text: "})
    # result = llm((f"""Your job is to reformat given data into nice looking format in markdown, only if it can be cultivated. Input:{result}.\n Reformated Markdown Text: """))
    st.write("Answer:", result["result"])


