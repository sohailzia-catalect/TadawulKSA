import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import base64

st.set_page_config(page_title="KSA XChange Agent", page_icon="🤖")
st.header("🤖 Saudi Exchange VAI Assistant", divider='rainbow')

openai.api_key = "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"

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


filename = r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data"


def split_texts(text, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)

    return splits


main_bg = "images/darken_image.jpg"
main_bg_ext = ".jpg"


def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64(main_bg)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/jpg;base64,{img}");
background-size: 100%;
background-position: center;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/jpg;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}


[data-testid="stToolbar"] {{
right: 2rem;

}}
.stTextInput {{
      position: fixed;
      bottom: 3rem;
      background: rgba(100,0,0,0);
    }}

</style>

"""

st.markdown(page_bg_img, unsafe_allow_html=True)

hide_st_style = """
 <style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
 </style>

"""

st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache_resource(ttl="1h")
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


os.environ["OPENAI_API_KEY"] = "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"

retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k", temperature=0.4, streaming=True)

qa_chain = RetrievalQA.from_llm(llm, retriever=retriever, verbose=True)

if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

print(len(msgs.messages))

avatars = {"human": "user", "ai": "assistant"}
avatar_emoji = {"human":"👳", "ai":"🤖"}

for msg in msgs.messages:
    st.chat_message(avatars[msg.type], avatar=avatar_emoji[msg.type]).write(msg.content)

if user_query := st.text_input(label="Ask me stuff", label_visibility="collapsed"):
    st.chat_message("user", avatar="👳").write(user_query)
    msgs.add_user_message(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[stream_handler])
        msgs.add_ai_message(response)