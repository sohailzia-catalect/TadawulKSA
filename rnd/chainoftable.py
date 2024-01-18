from llama_index.llama_pack import download_llama_pack
import dotenv
import pandas as pd
import os

download_llama_pack(
    "ChainOfTablePack",
    "./chain_of_table_pack",
    skip_load=True,
    # leave the below line commented out if using the notebook on main
    # llama_hub_url="https://raw.githubusercontent.com/run-llama/llama-hub/jerry/add_chain_of_table/llama_hub"
)
from chain_of_table_pack.base import ChainOfTableQueryEngine, serialize_table

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-4-1106-preview")

dotenv.load_dotenv("../.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("../Combined Data.csv")

query_engine = ChainOfTableQueryEngine(df, llm=llm, verbose=True)

response = query_engine.query("How many records are there?")

print(response.response)