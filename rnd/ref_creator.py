import os
import PyPDF2
import dotenv
import openai
import numpy as np
import tiktoken
from openai import OpenAI
import pandas as pd
import csv

dotenv.load_dotenv("../.env")

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_refs = {
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


for folder in os.listdir(r"/data"):

    pdf_reader = PyPDF2.PdfReader(os.path.join(r"/data", folder))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        pdf_refs[folder]["content"] = text
        response_output = client.embeddings.create(input=truncate_text_tokens(text), model="text-embedding-ada-002")
        embedding = response_output.data[0].embedding
        pdf_refs[folder]["embedding"] = embedding


# convert to proper format and then save as csv. The above code could also be modified instead of this. but fuck it
data = []
df = pd.DataFrame(pdf_refs)
for col in df.columns:
    data.append((col, df[col]["ref"], df[col]["embedding"], df[col]["content"]))
dff = pd.DataFrame(data, columns=["Name", "Ref", "Embedding", "Content"])
dff.to_pickle("references.pkl")

