# Import necessary libraries
import openai
import streamlit as st
import time
import os
import dotenv
from openai import OpenAI

# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="KSAXchange App", page_icon=":speech_balloon:")



dotenv.load_dotenv('.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_assistant = client.beta.assistants.create(
    name="OpenAI Rag",
    instructions=(
        " You are Saudi Exchange Assisstant.Your job is to provide answers to questions by considering the information given to you.Make sure to retrieve correct data and answer correctly.Do not answer irrelevant questions. Use your knowledge base to best respond to questions. "
        "NO MATTER WHAT, DO NOT PULL INFORMATION FROM EXTERNAL KNOWLEDGE. ONLY USE YOUR OWN KNOWLEDGE BASE. Present the answer in a nice format as well."
    ),
    model="gpt-4-1106-preview",
    tools=[{"type": "retrieval"}]
)

openai_assistant_id = "asst_UxxUZWWzzUZTT3xcdgCQhqtm"

thread_id = "thread_z39daLLWHnj9Nqu1NvZBW1Mp"

openai_assistant_id = openai_assistant.id

st.write(openai_assistant_id)

# Initialize session state variables for file IDs and chat control
if "file_id_list" not in st.session_state:
    st.session_state.file_id_list = []

if "thread_id" not in st.session_state:
    st.session_state.code_interpreter_thread_id = None

if "filed_added" not in st.session_state:
    st.session_state.filed_added = False


def upload_to_openai(filepath):
    """Upload a file to OpenAI and return its file ID."""
    with open(filepath, "rb") as file:
        response = openai.files.create(file=file.read(), purpose="assistants")
    return response.id


if not st.session_state.file_id_list:

    for folder in os.listdir(r"C:\Users\Catalect\PycharmProjects\tadawulchat\data"):
        additional_file_id = upload_to_openai(
            os.path.join(r"C:\Users\Catalect\PycharmProjects\tadawulchat\data", folder))
        st.session_state.file_id_list.append(additional_file_id)


# Display all file IDs
if not st.session_state.filed_added and st.session_state.file_id_list:
    st.sidebar.write("Uploaded File IDs:")
    for file_id in st.session_state.file_id_list:
        st.sidebar.write(file_id)
        # Associate files with the assistant
        assistant_file = client.beta.assistants.files.create(
            assistant_id=openai_assistant_id,
            file_id=file_id
        )
    st.session_state.filed_added = True

if st.session_state.code_interpreter_thread_id is None:
    thread = client.beta.threads.create()
    st.session_state.code_interpreter_thread_id = thread.id
    st.write(thread.id)

    st.write("executed")


# Main chat interface setup
st.title("KSA Xchange")

# Initialize the model and messages list if not already in session state
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-4-1106-preview"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for the user
if prompt := st.chat_input("What is up?"):
    # Add user message to the state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add the user's message to the existing thread
    client.beta.threads.messages.create(
        thread_id=st.session_state.code_interpreter_thread_id,
        role="user",
        content=prompt
    )

    # Create a run with additional instructions
    run = client.beta.threads.runs.create(
        thread_id=st.session_state.code_interpreter_thread_id,
        assistant_id=openai_assistant_id,
        instructions="Please answer the question using the knowledge. Make sure to present it in a nice and simple markdown where possible. "
    )

    # Poll for the run to complete and retrieve the assistant's messages
    while run.status != 'completed':
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.code_interpreter_thread_id,
            run_id=run.id
        )

    # Retrieve messages added by the assistant
    messages = client.beta.threads.messages.list(thread_id=st.session_state.code_interpreter_thread_id)

    # Process and display assistant messages
    assistant_messages_for_run = [
        message for message in messages
        if message.run_id == run.id and message.role == "assistant"
    ]
    for message in assistant_messages_for_run:
        full_response = message.content[0].text.value
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response, unsafe_allow_html=True)
