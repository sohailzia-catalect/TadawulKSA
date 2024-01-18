import streamlit as st

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangChainOpenAI
import os
import dotenv
import time
from openai import OpenAI
from llama_index.llms import OpenAI as LlamaIndexOpenAI
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    StorageContext,
    load_index_from_storage,
    Prompt,
    ServiceContext
)
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.retrievers import BM25Retriever
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
import numpy as np
import tiktoken
import pandas as pd
import re
from llama_index.llms import OpenAILike
from llama_index.embeddings.cohereai import CohereEmbedding

st.set_page_config(page_title="KSAXchange App", page_icon=":speech_balloon:")

dotenv.load_dotenv(".env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.sidebar.title("Chat Configs")

lang_choice = st.sidebar.radio("**Choose language**",
                               ["*:violet[العربية]*", "*:violet[English]*"])

option = st.sidebar.selectbox("What are you up to?",
                              ["GeneralInformationQA", "CompanyInfoQA",
                               "GeneralInformationQACustomImplementation", "ChatWithTabularDataCodeInterpreter",
                               "FastGeneralInfoQACustom"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

code_interpreter_thread_id = 'thread_lpwLrEN62PYe8x39jONuCaBn'
code_interpreter_openai_assistant_id = 'asst_BFbqgYh64qTlpGiiUAjBbgkE'
retrieval_openai_assistant_id = "asst_UxxUZWWzzUZTT3xcdgCQhqtm"
retrieval_thread_id = "thread_z39daLLWHnj9Nqu1NvZBW1Mp"

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

data_path = r"C:\Users\Catalect\PycharmProjects\tadawulchat\data"

template = (
    "You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you. "
    "\n Make sure to be helpful and only consider the information provided to you. "
    " \n Where possible, also present the information in a nice format. "
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question while providing as much details as possible and also present it in a nice markdown format: {query_str}\n"
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


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # whatever and how many retrievers u use, get their nodes and combine it.

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


@st.cache_resource
def get_re_ranker():
    # from llama_index.postprocessor import SentenceTransformerRerank
    # reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

    cohere_rerank = CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=3)
    return cohere_rerank


@st.cache_data
def get_pkl_df():
    pkl_file_path = "data.pkl"
    df = pd.read_pickle(pkl_file_path)
    return df


def get_contextual_answer(user_question, context):
    prompT = (
        f"You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you. \n Make sure to be helpful and only consider the information provided to you. \n If you think the question is irrelevant, just ask the user to go away!\n Also please format it in a nice markdown.{context}. Given this information, please answer the question in details and present it in a nice markdown format. : {user_question}. \nAnswer: ")
    return get_llm(modelname="gpt-3.5-turbo-16k-0613").complete(prompT).text


@st.cache_resource
def get_query_engine():
    #         node_postprocessors=[get_re_ranker()],

    retriever, index = get_hybrid_retriever()
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        service_context=index.service_context,
        text_qa_template=qa_template)

    return query_engine


def get_llama_index_query_engine():
    index = get_vector_index()
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=4)

    return query_engine


def chat_fast_llama_index(question):
    answer = get_llama_index_query_engine().query(question).response

    response_output = client.embeddings.create(input=truncate_text_tokens(answer),
                                               model="text-embedding-ada-002")

    df = get_pkl_df()
    embedding = response_output.data[0].embedding
    df["Simscores"] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    results = df.sort_values("Simscores", ascending=False)

    return answer + "\n\nFor more information, go to: " + results.reset_index()["Reference"][0]


def chat_general_info_custom_implementation(question):
    query_engine = get_query_engine()

    answer = query_engine.query(question).response

    response_output = client.embeddings.create(input=truncate_text_tokens(answer),
                                               model="text-embedding-ada-002")

    df = get_pkl_df()
    embedding = response_output.data[0].embedding
    df["Simscores"] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    results = df.sort_values("Simscores", ascending=False)

    context = results["Content"].iloc[0]
    answer = get_contextual_answer(question, context)

    return st.success(answer + "\n\nFor more information, go to: " + results.reset_index()["Reference"][0])


keywords = ["rajhi", "bilad", "saudi telecom", "telecom", "stc", "aramco", "saudiaramco", "sauditelecom", "alrajhi",
            "albilad", "acwapower", "acwa", "acwa power"]


def get_context():
    with open("company_info_context.txt", 'r') as file:
        # Read the entire contents of the file
        file_contents = file.read()

    return file_contents


def check_keywords(question):
    """
    Check if the given question contains any of the specified keywords.
    """
    question_lower = question.lower()

    for keyword in keywords:
        if keyword.lower() in question_lower:
            return True

    return False


def manual_answering(query):
    if not check_keywords(query):
        return "Sorry. I don't think I can help you with that. I have only learned information about the following companies: Al Bilad, Al Rajhi, Saudi Telecom, Saudi Aramco, ACWA Power. If you have any questions related to these companies, I will be happy to assist."
    else:
        context = get_context()
        prompt = f"You are Saudi Stock Exchange chat assistant. Your job is to provide information related to companies using the given context. Make sure to be helpful and only consider the information provided to you: {context}. \n Given this information, please answer: {query}"
        return get_llm().complete(prompt).text


@st.cache_resource
def get_hybrid_retriever():
    vector_index = get_vector_index()
    vector_based_retriever = vector_index.as_retriever()
    keyword_based_retriever = get_keyword_based_retriever(vector_index)
    hybrid_retriever = HybridRetriever(vector_based_retriever, keyword_based_retriever)
    return hybrid_retriever, vector_index


def get_keyword_based_retriever(index):
    service_context = ServiceContext.from_service_context(index.service_context)
    node_parser = service_context.node_parser
    nodes = node_parser.get_nodes_from_documents(get_data())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    return bm25_retriever


@st.cache_resource
def get_chat_tabular_qa_chain():
    loader = CSVLoader(file_path=r"C:\Users\Catalect\PycharmProjects\tadawulchat\Untitled spreadsheet - Cumulative.csv")
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    chain = RetrievalQA.from_chain_type(llm=LangChainOpenAI(model_name="gpt-4"), chain_type="stuff",
                                        retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    return chain


def get_llm(modelname="gpt-4-1106-preview"):
    return LlamaIndexOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=modelname, temperature=0.4, chunk_size=2048,
                            chunk_overlap=256)


# "mistralai/Mistral-7B-Instruct-v0.2"

# mistralai/Mixtral-8x7B-Instruct-v0.1

# togethercomputer/LLaMA-2-7B-32K


@st.cache_resource
def get_mistral_llm():
    return OpenAILike(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        api_base="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY"),
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0.1,
    )


def get_vector_index():
    service_context = ServiceContext.from_defaults(llm=get_llm("gpt-3.5-turbo-16k"))
    index = VectorStoreIndex.from_documents(get_data(), service_context=service_context)
    index.storage_context.persist("index/website_data_cohere_index_fastllamaindex")
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="index/website_data_cohere_index_fastllamaindex"),
        service_context=service_context)
    return index


def chat_tabular(question):
    qa_chain = get_chat_tabular_qa_chain()
    return qa_chain({"question": question})['result']


def translate_to_arabic(text):
    llm = get_llm("gpt-3.5-turbo-16k")
    return llm.complete(
        f"Translate this to Arabic. Keep the structure intact. Input: {text}.\n Arabic Translated Text:")


def chat_tabular_codeinterpreter(question):
    client.beta.threads.messages.create(
        thread_id=code_interpreter_thread_id,
        role="user",
        content=question)

    run = client.beta.threads.runs.create(
        thread_id=code_interpreter_thread_id,
        assistant_id=code_interpreter_openai_assistant_id,
    )

    while run.status != 'completed':
        run = client.beta.threads.runs.retrieve(
            thread_id=code_interpreter_thread_id,
            run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=code_interpreter_thread_id)

    assistant_messages_for_run = [
        message for message in messages
        if message.run_id == run.id and message.role == "assistant"]

    text_msgs = ""

    if len(assistant_messages_for_run) != 0:
        for run in assistant_messages_for_run:
            contents = run.content
            for item in contents:
                if item.type == "text":
                    text_msgs += f"\n{item.text.value}"

    if text_msgs.__eq__(""):
        return "Sorry. Not able to answer that. But I assure you that I will get better over time."

    llm = get_llm("gpt-3.5-turbo")
    refactored_response = llm.complete(
        f"Act like a helpful assistant. You are given a question and possible response to it. If there is some irrelevant bits in the response, disregard them and make the response concise and to the point. Questions: {question}. \n Response: {text_msgs}.\n Impproved Response: ")
    return refactored_response


def chat_general_info(question):
    client.beta.threads.messages.create(
        thread_id=retrieval_thread_id,
        role="user",
        content=question)

    start_time = time.time()

    run = client.beta.threads.runs.create(
        thread_id=retrieval_thread_id,
        assistant_id=retrieval_openai_assistant_id)

    while run.status != 'completed':
        run = client.beta.threads.runs.retrieve(
            thread_id=retrieval_thread_id,
            run_id=run.id)

    # Retrieve messages added by the assistant
    messages = client.beta.threads.messages.list(thread_id=retrieval_thread_id)

    # Process and display assistant messages
    assistant_messages_for_run = [
        message for message in messages
        if message.run_id == run.id and message.role == "assistant"]

    st.write("This took me about: " + str(time.time() - start_time))

    text_msgs = ""

    if len(assistant_messages_for_run) != 0:
        for run in assistant_messages_for_run:
            contents = run.content
            for item in contents:
                if item.type == "text":
                    text_msgs += f"\n{item.text.value}"

    if text_msgs.__eq__(""):
        return "Sorry. Not able to answer that. But I assure you that I will get better over time."

    regex_pattern = r"【.*?】"

    cleaned_string = re.sub(regex_pattern, '', text_msgs)

    return cleaned_string


if option == "ChatWithTabularDataLangChain":
    text_input = st.text_input("Example: Tell me which company has the highest mean High price average?...")
    if text_input:
        stt_time = time.time()
        response = chat_tabular(text_input)
        print("Inference time: " + str(time.time() - stt_time) + " seconds.")
        st.write(response)

elif option == "ChatWithTabularDataCodeInterpreter":
    text_input = st.text_input("Example: Tell me which company has the highest mean High price average?...")
    if text_input:
        stt_time = time.time()
        response = chat_tabular_codeinterpreter(text_input)
        print("Inference time: " + str(time.time() - stt_time) + " seconds.")
        if lang_choice.__eq__("*:violet[العربية]*"):
            response = translate_to_arabic(response)
        st.markdown(response, unsafe_allow_html=True)


elif option == "GeneralInformationQA":
    text_input = st.text_input("How may I help u?")
    if text_input:
        stt_time = time.time()
        response = chat_general_info(text_input)
        print("Inference time: " + str(time.time() - stt_time) + " seconds.")
        if lang_choice.__eq__("*:violet[العربية]*"):
            response = translate_to_arabic(response)
        st.markdown(response, unsafe_allow_html=True)


elif option == "GeneralInformationQACustomImplementation":
    text_input = st.text_input("How may I help u?")
    if text_input:
        stt_time = time.time()
        response = chat_general_info_custom_implementation(text_input)
        print("Inference time: " + str(time.time() - stt_time) + " seconds.")
        if lang_choice.__eq__("*:violet[العربية]*"):
            response = translate_to_arabic(response)
        st.markdown(response, unsafe_allow_html=True)


elif option == "CompanyInfoQA":
    text_input = st.text_input("How may I help u?")
    if text_input:
        stt_time = time.time()
        response = manual_answering(text_input)
        print("Inference time: " + str(time.time() - stt_time) + " seconds.")
        if lang_choice.__eq__("*:violet[العربية]*"):
            response = translate_to_arabic(response)
        st.markdown(response, unsafe_allow_html=True)


elif option == "FastGeneralInfoQACustom":
    text_input = st.text_input("How may I help u?")
    if text_input:
        stt_time = time.time()
        response = chat_fast_llama_index(text_input)
        print("Inference time: " + str(time.time() - stt_time) + " seconds.")
        if lang_choice.__eq__("*:violet[العربية]*"):
            response = translate_to_arabic(response)
        st.markdown(response, unsafe_allow_html=True)
