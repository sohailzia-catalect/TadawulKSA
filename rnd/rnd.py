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
from openai import OpenAI

dotenv.load_dotenv("../.env")

client = OpenAI()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

data = {
    "About Us.pdf": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/about-saudi-exchange/aboutus?locale=en"},

    "Become a Member.pdf": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/trading/participants-directory/become-a-member?locale=en"},

    "Become an Inf_Index Provider.pdf": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/trading/participants-directory/become-an-information-provider?locale=en"},

    "Issuer.pdf": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/listing/become-an-issuer?locale=en"},

    "Nomu - Parallel Market.pdf": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/rules-guidance/capital-market-overview/Equities?locale=en"},

    "Sukuk and Bonds.pdf": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/rules-guidance/capital-market-overview/sukuk-and-bonds?locale=en"},

    "Investor.pdf": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/trading/investing-trading/become_an_investor?locale=en" + "\n" +
               "https://www.saudiexchange.sa/wps/portal/saudiexchange/trading/investing-trading/qualified_foreign_investors"}
}


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


assistant_prompt = "You will be given some context related to Saudi Stock Exchange company. Your job is to tell what kind of information is present in the context as bullet points. Make sure to include all the necessary detials. "


def get_summarized_content_gpt4(content):
    message = [{"role": "assistant", "content": assistant_prompt},
               {"role": "user", "content": "Extract from given information: " + content}]
    temperature = 0.6

    response = client.chat.completions.create(
        model="gpt-4",
        messages=message,
        temperature=temperature,
    )
    return response.choices[0].message.content


def get_all_data():
    for folder in os.listdir(r"/data"):

        pdf_reader = PyPDF2.PdfReader(os.path.join(r"/data", folder))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        summarized_content = get_summarized_content_gpt4(text)
        response_output = client.embeddings.create(input=truncate_text_tokens(summarized_content),
                                                   model="text-embedding-ada-002")
        embedding = response_output.data[0].embedding

        data[folder]["content"] = summarized_content
        data[folder]["raw_content"] = text
        data[folder]["embedding"] = embedding
    df = pd.DataFrame(data)

    data_final = []

    for col in df.columns:
        data_final.append((col, df[col]["content"], df[col]["raw_content"], df[col]["embedding"], df[col]["ref"]))
    dff = pd.DataFrame(data_final, columns=["Name", "Content", "Raw Content", "Embedding", "Reference"])
    dff.to_pickle("data.pkl")
    dff.to_csv("data.csv")


get_all_data()


