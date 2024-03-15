import os
from typing import Optional, Any
from uuid import UUID
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
import os
import psutil

pid = os.getpid()
python_process = psutil.Process(pid)

dotenv.load_dotenv(".env")

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


filename = "data"

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


def get_pkl_df():
    pkl_file_path = "references.pkl"
    df = pd.read_pickle(pkl_file_path)
    return df


def get_qa_chain():
    retriever = configure_retriever()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k", temperature=0.4, streaming=True, api_key=os.getenv("OPENAI_API_KEY"))

    qa_chain = RetrievalQA.from_llm(llm, retriever=retriever, verbose=True)
    return qa_chain


avatars = {"human": "user", "ai": "assistant"}
avatar_emoji = {"human": "👳", "ai": "🤖"}

user_query = "how do i become a member?"

NUM_OF_CONCURRENT_REQUESTS = 100



import concurrent.futures
import psutil  # Make sure to install it using: pip install psutil


def run_qa_chain(user_query):
    response = get_qa_chain().run(user_query)
    ram_usage_after = measure_ram_usage()
    print(ram_usage_after)
    return response

def measure_ram_usage():
    process = psutil.Process()

    return process.memory_info().rss / (1024 ** 2)  # Convert to MB

def main():
    user_query = "how do i become a member?"

    # Set the number of parallel tasks (adjust as needed)
    num_parallel_tasks = 20

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_tasks) as executor:
        futures = [executor.submit(run_qa_chain, user_query) for _ in range(num_parallel_tasks)]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

        # Collect results (if needed)
        results = [future.result() for future in futures]

    # Measure RAM usage after completion
    ram_usage_after = measure_ram_usage()

    print(f"RAM usage before: {ram_usage_before:.2f} MB")
    print(f"RAM usage after: {ram_usage_after:.2f} MB")

if __name__ == "__main__":
    # Measure RAM usage before starting
    ram_usage_before = measure_ram_usage()

    # Run the main function
    main()

