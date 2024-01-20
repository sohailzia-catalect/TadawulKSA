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
    "About Us": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/about-saudi-exchange/aboutus?locale=en"},

    "Become a Member": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/trading/participants-directory/become-a-member?locale=en"},

    "Become an Inf_Index Provider": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/trading/participants-directory/become-an-information-provider?locale=en"},

    "Issuer": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/listing/become-an-issuer?locale=en"},

    "Nomu - Parallel Market": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/rules-guidance/capital-market-overview/Equities?locale=en"},

    "Sukuk and Bonds": {
        "ref": "https://www.saudiexchange.sa/wps/portal/saudiexchange/rules-guidance/capital-market-overview/sukuk-and-bonds?locale=en"},

    "Investor": {
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


def get_context(folder):
    with open(folder, 'r') as file:
        # Read the entire contents of the file
        file_contents = file.read()

    return file_contents

def get_all_data():
    for folder in os.listdir(r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data"):
        file_ext = folder.split(".")[-1]
        name = folder.split(".")[0]
        if file_ext.__eq__("txt"):
            filecontents = get_context(os.path.join(r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data", folder))
            contents = filecontents.split("Company Name:")
            for content in contents[1:]:
                company_name = content.split("\n")[0]
                ref = content.split("Reference:")[-1].split("\n")[0]

                response_output = client.embeddings.create(input=truncate_text_tokens(content),
                                                           model="text-embedding-ada-002")
                embedding = response_output.data[0].embedding
                data[company_name] = {}
                data[company_name]["content"] = content
                data[company_name]["embedding"] = embedding
                data[company_name]["ref"] = ref

        else:
            pdf_reader = PyPDF2.PdfReader(os.path.join(r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data", folder))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            response_output = client.embeddings.create(input=truncate_text_tokens(text),
                                                       model="text-embedding-ada-002")
            embedding = response_output.data[0].embedding

            data[name]["content"] = text
            data[name]["embedding"] = embedding

    df = pd.DataFrame(data)

    data_final = []

    for col in df.columns:
        data_final.append((col, df[col]["content"], df[col]["embedding"], df[col]["ref"]))
    dff = pd.DataFrame(data_final, columns=["Name", "Content", "Embedding", "Reference"])
    dff.to_pickle("data_updated.pkl")
    dff.to_csv("data_updated.csv")


get_all_data()


