from dataclasses import dataclass
from typing import Literal
import streamlit as st
from pathlib import Path
import base64
from langchain.llms.openai import OpenAI
from langchain.callbacks import get_openai_callback
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
import streamlit.components.v1 as components
import os
import dotenv
import numpy as np
from langchain.llms import Together
import pandas as pd
from langchain.vectorstores import FAISS
import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
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
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.retrievers import BM25Retriever
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine

dotenv.load_dotenv(".env")

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

client = orig_openai(api_key=os.getenv("OPENAI_API_KEY"))

openai.api_key = os.getenv("OPENAI_API_KEY")

data_path = r"C:\Users\Catalect\PycharmProjects\tadawulchat\data"

template = (
    "You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you. "
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


def get_rag_llm_mistral():
    rag_llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
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


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
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
def get_vector_index():
    service_context = ServiceContext.from_defaults(
        llm=llm
    )
    # index = VectorStoreIndex.from_documents(get_data(), service_context=service_context)
    # index.storage_context.persist("index/website_data_index")
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir="index/website_data_index"))

    return index


import time


def get_keyword_based_retriever(index):
    service_context = ServiceContext.from_service_context(index.service_context)
    node_parser = service_context.node_parser
    nodes = node_parser.get_nodes_from_documents(get_data())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    return bm25_retriever


@st.cache_resource
def get_hybrid_retriever():
    vector_index = get_vector_index()
    st = time.time()
    vector_based_retriever = vector_index.as_retriever()
    print("Time taken for vector based: ", time.time() - st)
    keyword_based_retriever = get_keyword_based_retriever(vector_index)
    st = time.time()
    hybrid_retriever = HybridRetriever(vector_based_retriever, keyword_based_retriever)
    print("Time taken for keyword based: ", st)

    return hybrid_retriever, vector_index


@st.cache_resource
def get_re_ranker():
    # from llama_index.postprocessor import SentenceTransformerRerank
    # reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

    cohere_rerank = CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=3)
    return cohere_rerank


@st.cache_data
def get_query_engine():
    service_context = ServiceContext.from_defaults(llm=llm)

    index = VectorStoreIndex.from_documents(get_data(), service_context=service_context)
    index.storage_context.persist("index/website_data_index")
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir="index/website_data_index"))

    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)
    return query_engine


def get_query_engine_2():
    retriever, index = get_hybrid_retriever()
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        service_context=index.service_context,
        text_qa_template=qa_template)
    return query_engine


def get_response(question):
    query_engine = get_query_engine_2()

    response = query_engine.query(question)
    return response


@st.cache_data
def get_pkl_df():
    pkl_file_path = "data.pkl"
    df = pd.read_pickle(pkl_file_path)
    return df


def get_contextual_answer(user_question, context):
    prompT = (
        f"You are Saudi Stock Exchange chat assistant. Your job is to answer queries with information provided to you. "
        f"\n Make sure to be helpful and only consider the information provided to you."
        f"\n If you think the question is irrelevant, just ask the user to go away!"
        "---------------------\n"
        f"{context}"
        "\n---------------------\n"
        f"Given this information, please answer the question: {user_question}. \nAnswer: ")
    return get_rag_llm()(prompT)


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str


def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def responser(user_question):
    answer = get_response(user_question).response

    response_output = client.embeddings.create(input=truncate_text_tokens(answer),
                                               model="text-embedding-ada-002")
    df = get_pkl_df()
    embedding = response_output.data[0].embedding
    df["Simscores"] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    results = df.sort_values("Simscores", ascending=False)

    context = results["Raw Content"].iloc[0]
    answer = get_contextual_answer(user_question, context)
    # if len(answer) > 300:
    #     answer = get_rag_llm_mistral()((
    #                                         f"Your job is to reformat given data into nice looking format in markdown, only if it can be cultivated. Input:{answer}.\n Reformated Markdown Text: "))
    # return (answer + "\n\n" + "\n For more detailed information, go to: " + results["Reference"][0] )
    return answer

            # + "\n\n I was able find relevant data with conf score of " + str(results["Simscores"][0]))


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        st.session_state.conversation = responser


def on_click_callback():
    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt
        llm_response = st.session_state.conversation(
            human_prompt
        )

        st.session_state.history.append(
            Message("human", human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response)
        )
        st.session_state.token_count += cb.total_tokens


load_css()
initialize_session_state()

st.image("images/second_image.jpg", width=700)
st.title("SaudiXChange Helpdesk 🤖")

file_ = open("images/ksa icon cropped.png", "rb")
contents = file_.read()
ai_icon_url = base64.b64encode(contents).decode("utf-8")
file_.close()

file_ = open("images/user_icon.png", "rb")
contents = file_.read()
user_icon_url = base64.b64encode(contents).decode("utf-8")
file_.close()

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row 
            {'' if chat.origin == 'ai' else 'row-reverse'}">
            <img class="chat-icon" src="data:image/png;base64,{ai_icon_url if chat.origin == 'ai' else user_icon_url}"
                 width=45 height=45>
            <div class="chat-bubble
            {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
                """
        st.markdown(div, unsafe_allow_html=True)

    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="Thou shall ask...",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )

components.html("""
<script>
const streamlitDoc = window.parent.document;

const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);

streamlitDoc.addEventListener('keydown', function(e) {
    switch (e.key) {
        case 'Enter':
            submitButton.click();
            break;
    }
});
</script>
""",
                height=0,
                width=0,
                )
