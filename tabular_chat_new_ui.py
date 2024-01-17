from dataclasses import dataclass
from typing import Literal
import base64
from langchain.callbacks import get_openai_callback
import streamlit as st
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

from openai import OpenAI as orig_openai
import streamlit.components.v1 as components
import os
import dotenv
from langchain.llms import Together
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
import openai
from llama_index import (
    Prompt,
)

from llama_index.retrievers import BaseRetriever
from langchain import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

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


llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.1,
    max_tokens=512,
    top_k=0.5,
    together_api_key=os.getenv("TOGETHER_API_KEY")
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


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str


def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


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

retriever = vectorstore.as_retriever(k=1)

template = """You are a stock exchange analyst for Saudi Exchange. Use the following context to answer a question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible and don't make up own facts. Think properly before you reply. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""


@st.cache_resource
def get_qa_chain():
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Run chain

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=ensemble_retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain


def responser(user_question):
    qa_chain = get_qa_chain()
    result = qa_chain({"query": user_question})
    result = result["result"]
    result = qa_chain({
        "query": f"Your job is to reformat given data into nice looking format in markdown. Input:{result}.\n Reformated Markdown Text: "})
    # result = llm((f"""Your job is to reformat given data into nice looking format in markdown, only if it can be cultivated. Input:{result}.\n Reformated Markdown Text: """))
    return result["result"]


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
