# Import necessary libraries
import openai
import time
import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv('.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

file_id_list = []

thread = client.beta.threads.create()
thread_id = 'thread_lpwLrEN62PYe8x39jONuCaBn'



def upload_to_openai(filepath):
    with open(filepath, "rb") as file:
        response = openai.files.create(file=file.read(), purpose="assistants")
    return response.id


# if len(file_id_list) == 0:
#     additional_file_id = upload_to_openai(f"Combined Data.csv")
#     file_id_list.append(additional_file_id)
#
# openai_assistant = client.beta.assistants.create(
#     name="OpenAI Code Interpreter",
#     instructions=(
#         "You are a personal Data Analyst Assistant. You have been provided data for 5 different companies. Your job is to answer questions using the data provided to you. Be helpful. If you are not able answer, just say that you don't know. "),
#     model="gpt-4-1106-preview",
#     tools=[{"type": "code_interpreter"}],
#     file_ids=file_id_list)

openai_assistant_id = 'asst_BFbqgYh64qTlpGiiUAjBbgkE'

prompt = "How many records are there?"

client.beta.threads.messages.create(
    thread_id=thread_id,
    role="user",
    content=prompt
)

run = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=openai_assistant_id,
    instructions="Please answer the queries and responsd like an assistant. Be helpful."

)

while run.status != 'completed':
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run.id)

messages = client.beta.threads.messages.list(thread_id=thread_id)

assistant_messages_for_run = [
    message for message in messages
    if message.run_id == run.id and message.role == "assistant"
]

text_msgs = ""
image_msgs = []


def download_images():
    num_of_imgs_downloaded = 0
    for idx, file_id in enumerate(image_msgs):
        api_response = client.files.with_raw_response.retrieve_content(file_id)
        if api_response.status_code == 200:
            content = api_response.content
            with open(f'image-{idx}.png', 'wb') as f:
                f.write(content)
                num_of_imgs_downloaded += 1
    return num_of_imgs_downloaded


if len(assistant_messages_for_run) != 0:
    contents = assistant_messages_for_run[0].content
    for item in contents:
        if item.type == "image_file":
            image_msgs.append(item.image_file.file_id)
        elif item.type == "text":
            text_msgs += f"\n{item.text}"

