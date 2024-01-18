import openai
import os
import PyPDF2
import random
import itertools
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
import os
import glob


openai.api_key = "<Openai_key>"

os.environ["OPENAI_API_KEY"] = ""

def load_docs(files):
    all_text = ""
    for file_path in files:

        pdf_reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        all_text += text

    return all_text


def get_retriever(filename):
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.load_local(filename, embeddings)
    return retriever.as_retriever()


def folder_available(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    folder_path += "_index"
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

filename = ""

def create_retriever(_embeddings):
    loaded_text = load_docs(filename)

    # Split the document into chunks
    splits = split_texts(loaded_text, chunk_size=1500,
                         overlap=200)

    # Display the number of text chunks
    num_chunks = len(splits)

    vectorstore = FAISS.from_texts(splits, _embeddings)
    retriever = vectorstore.as_retriever(k=5)
    # vectorstore.save_local(r'C:\Users\Catalect\PycharmProjects\OthmanFinProject\RnD\{}_index'.format(ony_filename))
    return retriever


def split_texts(text, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)

    return splits


user_question = ""


def main():
    embeddings = OpenAIEmbeddings()

    retriever = create_retriever(embeddings)

    chat_openai = ChatOpenAI(model_name="gpt-4", verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

    if user_question:
        answer = qa.run(user_question)


if __name__ == "__main__":
    main()
