import os
import time

import dotenv
import PyPDF2
import openai
from openai import OpenAI

client = OpenAI(api_key="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")

dotenv.load_dotenv('.env')

openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

context = ""

quantity = 20

prompt = f"Given the following context: {context}\n\n----------------\nCreate {quantity} detailed question-answer pairs from the context above, the questions are asked by a curious user on a Saudi Stock Exchange Website and the answers are by a helpful Saudi Stock Exchange AI-Assistant. The question-answer pairs should abide by the following rules: \n1. All question/answer pairs have to be strictly based on the context provided above and be self-contained and independent.\n2. The questions should be diverse and cover different aspects of the context provided above.\n3. The answers should be long, extensive, detailed, informative, helpful and self-contained. Return me in a json format with question and answer as keys."

data_path = r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data"


def get_response_openai(message):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=message,
        temperature=0.4,
        max_tokens=32000,
    )
    return response


for file in os.listdir(data_path):
    pdf_reader = PyPDF2.PdfReader(
        os.path.join(r"C:\Users\Catalect\Documents\GitHub\TadawulKSA\data", file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    context = text
    prompt = f"Given the following context: {context}\n\n----------------\nCreate {quantity} detailed question-answer pairs from the context above, the questions are asked by a curious user on a Saudi Stock Exchange Website and the answers are by a helpful Saudi Stock Exchange AI-Assistant. The question-answer pairs should abide by the following rules: \n1. All question/answer pairs have to be strictly based on the context provided above and be self-contained and independent.\n2. The questions should be diverse and cover different aspects of the context provided above.\n3. The answers should be long, extensive, detailed, informative, helpful and self-contained."

    res = get_response_openai([{
        "role": "system",
        "content": f"You are generating data which will be used to train a Questing Answering model related to Saudi Stock Exchange Website.\n\Make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model. Return the data in a json format with question and answer as keys\n`{prompt}`"
    },
        {"role": "user", "content": prompt}])

    print(res)

