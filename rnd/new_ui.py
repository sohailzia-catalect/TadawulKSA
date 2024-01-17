from dataclasses import dataclass
from typing import Literal
import streamlit as st
from pathlib import Path
import base64
from langchain.llms.openai import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.conversational_retrieval.base import ChatVectorDBChain
from langchain.chains import RetrievalQA
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

dotenv.load_dotenv("../.env")

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = FAISS.load_local(r"C:\Users\Catalect\PycharmProjects\tadawulchat\index_storage", embeddings)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the dominant stock market  in Gulf region, Saudi Exchange.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about general information related to Saudi Exchange, stock market .
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about Saudi Exchange stock market, politely inform them that you are tuned to only answer questions about it.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@st.cache_data
def get_pkl_df():
    pkl_file_path = r"/rnd/references.pkl"
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


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str


def load_css():
    with open("../static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.1,
            max_tokens=512,
            top_k=0.5,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )
        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .1})
        st.session_state.conversation = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                                                    verbose=True)


def on_click_callback():
    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt
        llm_response = st.session_state.conversation.run(
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

st.image("second_image.jpg", width=700)
st.title("SaudiXChange Helpdesk 🤖")

file_ = open("../images/ksa icon cropped.png", "rb")
contents = file_.read()
ai_icon_url = base64.b64encode(contents).decode("utf-8")
file_.close()

file_ = open("../images/user_icon.png", "rb")
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
        value="Thou shall ask ya حمار",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )


# credit_card_placeholder.caption(f"""
# Used {st.session_state.token_count} tokens \n
# # Debug Langchain conversation:
# # st.session_state.conversation.memory.buffer
# # """)

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
