from pandasai import SmartDataframe
import streamlit as st
from pandasai import Agent

import pandas as pd

from pandasai.llm import OpenAI

llm = OpenAI(api_token="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")

df = pd.read_csv(r"C:\Users\Catalect\PycharmProjects\tadawulchat\Combined Data.csv")

gent = Agent(
    [df],
    config={
        "llm": OpenAI(api_token="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"),
        "verbose": True,
        "enforce_privacy": True,
        "enable_cache": True,
        "conversational": False,
        "save_charts": True,  # Or True. Both work as expected
        "open_charts": False,
    },
)
user_question = st.text_input("Enter your question:")

if user_question:
    response = gent.chat(user_question, output_type="plot")
    st.write(response)

